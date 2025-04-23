from __future__ import annotations

import asyncio
import atexit
import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from functools import partial, wraps
from pathlib import Path
from typing import Any, Concatenate, ParamSpec, Self, TypeVar, assert_never, overload

import torch
import torch.distributed as dist
import tree
from dask import config
from dask.distributed import Actor, ActorFuture, Client
from distributed.utils import sync

try:
    assert torch.__version__ >= "2.6.0", "FSDP2 requires PyTorch 2.6.0 or higher"
    from torch.distributed.fsdp import (
        CPUOffloadPolicy,
        MixedPrecisionPolicy,
        OffloadPolicy,
        fully_shard,
        register_fsdp_forward_method,
    )
except (ImportError, AssertionError) as e:
    raise ImportError(f"FSDP2 requires PyTorch 2.6.0 or higher: {e}") from e
from transformers import PreTrainedModel
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.tokenization_utils_base import BatchEncoding

from ldp.graph.async_torch import AsyncBufferedWorker, AsyncTorchModule
from ldp.nn.handlers.chunking import TensorChunker
from ldp.nn.handlers.module_handler import ModuleExecutionInterface, ModuleHandler
from ldp.nn.lm_config import TorchDType
from ldp.nn.utils import set_seed

from .transformer_handler import (
    AsyncTransformer,
    ExecutionMode,
    FSDPConfig,
    LMType,
    ParallelModeConfig,
    ParallelWorkerConfig,
    TransformerHandlerConfig,
    _get_data_device,
    _get_tokenized_inputs,
    _move_tensor,
    _process_outputs,
    maybe_set_tokenizer_chat_template,
)

logger = logging.getLogger(__name__)

config.set({
    # We have no use for rebooting workers in aviary for now, and rebooting workers
    # is annoying when debugging.
    "distributed.scheduler.allowed-failures": 0,
    # FSDP forward/backward passes can take way longer than the default warning at 3s
    "distributed.admin.tick.limit": "30s",
    # Gives us more time to debug a downed worker. TODO: see if there are negative consequences
    # of having this always enabled
    "distributed.comm.timeouts.connect": "300s",
    "distributed.comm.timeouts.tcp": "1200s",
})

TReturn = TypeVar("TReturn")
TParams = ParamSpec("TParams")


class AsyncTransformerInterface(ModuleExecutionInterface, AsyncTorchModule, ABC):
    """Base class for async interactions with a transformer model."""

    @abstractmethod
    async def __call__(  # type: ignore[override]
        self,
        inputs: str | BatchEncoding | list[dict],
        tools_json: list[dict] | None = None,
        **kwargs,
    ) -> tuple[str, torch.Tensor]:
        """Call the transformer on a single input, which may be encoded."""

    @staticmethod
    def model_generate(model: PreTrainedModel, *args, **kwargs):
        """A method that can be used as module_call_fn to sample from an LLM."""
        logger.debug(
            f"model.generate() input_ids shape: {kwargs['input_ids'].shape}, rank"
            f" {os.environ.get('RANK')}"
        )
        return model.generate(
            *args,
            **kwargs,
            pad_token_id=model.config.pad_token_id,  # not always set properly by .generate()
            eos_token_id=model.config.eos_token_id,
        )


class TransformerHandler(ModuleHandler):
    def __init__(self, config: TransformerHandlerConfig):
        # Maybe this should be configurable? Hard to isolate the effect though
        torch.set_float32_matmul_precision("high")

        self.config = config
        # Use local_rank to resolve model location only in the main process *for each node*
        config.lm_config.resolve_model_location(is_main_process=self.local_rank == 0)

        match config.lm_type:
            case LMType.GENERATION:
                tokenizer, model = config.lm_config.get_causal_lm()
                # On left for https://github.com/huggingface/transformers/pull/7552
                # ^ that seems to work for most HF models w/ absolute position embeddings
                # Left padding always works for relative position embeddings
                tokenizer.padding_side = "left"
            case LMType.REGRESSION:
                tokenizer, model = config.lm_config.get_regression_lm()
            case _:
                assert_never(config.lm_type)
        super().__init__(model)
        self.tokenizer = tokenizer
        maybe_set_tokenizer_chat_template(
            self.tokenizer, self.config.lm_config.chat_template
        )

        self._setup_fsdp()

        if config.checkpoint is not None:
            self.load_checkpoint(config.checkpoint)

    def _setup_fsdp(self):
        """Set up FSDP2 module wrapping."""
        if not dist.is_initialized():
            # For single device usage, just move to device directly
            device = self.config.lm_config.device
            self.module = self.module.to(device)
            return

        # Setup mixed precision policy if needed
        mp_policy = None
        if self.config.lm_config.dtype == TorchDType.bf16:
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                output_dtype=torch.bfloat16,
            )

        # Apply FSDP2 wrapping using new API
        fsdp_config = self.config.parallel_mode_config or FSDPConfig()
        offload_policy = (
            CPUOffloadPolicy(pin_memory=True)
            if fsdp_config.offload_cpu
            else OffloadPolicy()
        )

        self.module = fully_shard(
            self.module,
            mesh=None,  # Maybe we activate it later, see https://pytorch.org/docs/stable/distributed.html#torch.distributed.device_mesh.DeviceMesh
            reshard_after_forward=fsdp_config.reshard_after_forward,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
        )

        # Register model.generate as an FSDP forward method to handle generation correctly
        register_fsdp_forward_method(self.module, "generate")

    @property
    def local_rank(self) -> int:
        return int(os.getenv("LOCAL_RANK", "0"))

    def load_checkpoint(self, ckpt: os.PathLike | str, **kwargs) -> None:
        logger.info(f'Loading checkpoint from "{ckpt}"')
        start_time = time.perf_counter()
        ckpt_path = Path(ckpt)
        if ckpt_path.is_dir():
            # Assume it's a directory containing sharded state dict
            state_dict = torch.load(
                ckpt_path / f"rank{self.local_rank}_checkpoint.pt", map_location="cpu"
            )
            self.module.load_state_dict(state_dict)
        else:
            # Assume it's a single file containing a full state dict
            state_dict = torch.load(ckpt, map_location="cpu")
            # Load the state dict - will automatically handle the sharding
            self.module.load_state_dict(state_dict)

        self.barrier()
        logger.info(f"Loading checkpoint took {time.perf_counter() - start_time:.2f}s")
        torch.cuda.empty_cache()

    def save_checkpoint(self, ckpt: os.PathLike | str, **kwargs) -> None:
        start_time = time.perf_counter()
        ckpt_path = Path(ckpt)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        state_dict = self.module.state_dict()
        if dist.is_initialized():
            torch.save(state_dict, ckpt_path / f"rank{self.local_rank}_checkpoint.pt")
        else:
            torch.save(state_dict, ckpt_path / "checkpoint.pt")

        self.barrier()
        logger.info(f"Saving checkpoint took {time.perf_counter() - start_time:.2f}s")

    @staticmethod
    def barrier() -> None:
        if dist.is_initialized():
            dist.barrier()


class ParallelTransformerHandler(TransformerHandler):
    def __init__(
        self,
        config: TransformerHandlerConfig,
        parallel_worker_config: ParallelWorkerConfig,
    ):
        parallel_worker_config.set_env_vars()
        dist.init_process_group(
            backend="nccl", device_id=torch.device(f"cuda:{self.local_rank}")
        )
        dist.barrier()
        torch.cuda.set_device(self.local_rank)
        dist.barrier()
        self.worker_config = parallel_worker_config
        super().__init__(config)

    def set_seed(self, seed: int) -> None:
        """Set the seed for the current worker."""
        set_seed(seed)

    def _exec_func(
        self,
        func: Callable[Concatenate[Self, TParams], TReturn] | str,
        *args,
        **kwargs,
    ) -> TReturn:
        # data will be on CPU when sent from controller
        data_device = _get_data_device()
        to_device = partial(_move_tensor, device=data_device)
        args = tree.map_structure(to_device, args)
        kwargs = tree.map_structure(to_device, kwargs)

        try:
            with torch.autocast(
                device_type=self.module.device.type,
                dtype=torch.bfloat16
                if self.config.lm_config.dtype == TorchDType.bf16
                else torch.float32,
            ):
                res = (
                    getattr(self, func)(*args, **kwargs)
                    if isinstance(func, str)
                    else func(self, *args, **kwargs)
                )

            # Needed to prevent GPU memory leak to the main process scheduling the workers
            if isinstance(res, GenerateDecoderOnlyOutput):
                res.past_key_values = None
                res["past_key_values"] = None

            to_cpu = partial(_move_tensor, device=torch.device("cpu"))
            return tree.map_structure(to_cpu, res)
        except Exception as e:
            # Re-raise the exception with traceback preserved. For some exceptions, Dask
            # modifies or loses the original traceback when crossing process boundaries.
            # RuntimeError preserves the traceback when using with_traceback() of original
            # exception.
            raise RuntimeError(str(e)).with_traceback(e.__traceback__)  # noqa: B904

    def __del__(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()


class ParallelAsyncTransformer(AsyncTransformerInterface):
    def __init__(self, config: TransformerHandlerConfig):
        self._initialized = False

        parallel_mode_config = config.parallel_mode_config
        if not parallel_mode_config:
            raise ValueError("Parallel mode config must be provided.")
        self.config = config
        self.tokenizer = config.lm_config.get_tokenizer()
        maybe_set_tokenizer_chat_template(
            self.tokenizer, self.config.lm_config.chat_template
        )

        match parallel_mode_config.execution_mode:
            # TODO: see if we can just access `parallel_mode_config` as a
            # `config` attribute instead of passing both.
            case ExecutionMode.LOCAL_MACHINE:
                self._init_local_cluster(config, parallel_mode_config)
            case ExecutionMode.SLURM_CLUSTER:
                self._init_slurm_cluster(config, parallel_mode_config)
            case _:
                assert_never(parallel_mode_config.execution_mode)

        self._initialized = True

        atexit.register(self.teardown)

        # don't call AsyncTorchModule.__init__ because we don't need to set up module[_call_fn]
        AsyncBufferedWorker.__init__(
            self,
            batch_size=config.batch_size,
            max_wait_interval=config.max_wait_interval,
            collate_fn=config.collate_fn,
            decollate_fn=config.decollate_fn,
        )

        def handler_call_fn(handler: ParallelTransformerHandler, *args, **kwargs):
            return config.module_call_fn(handler.module, *args, **kwargs)

        self.handler_call_fn = handler_call_fn

    def _init_local_cluster(
        self, config: TransformerHandlerConfig, parallel_mode_config: ParallelModeConfig
    ):
        """Initialize a Dask cluster on local machine."""
        # lazy import since dask-cuda only works on Linux machines
        from dask_cuda import LocalCUDACluster

        kwargs = {}
        if os.environ.get("USE_UCX"):
            kwargs = {
                "protocol": "ucx",
                "enable_tcp_over_ucx": True,
                "enable_infiniband": True,
                "enable_nvlink": True,
            }

        self.cluster = LocalCUDACluster(
            n_workers=parallel_mode_config.num_workers,
            threads_per_worker=parallel_mode_config.num_cpus_per_worker,
            host=parallel_mode_config.scheduler_addr,
            port=parallel_mode_config.scheduler_port,
            memory_limit=None,  # do not let Dask manage memory - if we OOM, we OOM
            device_memory_limit=0,  # Disable gpu memory spilling. Should be handled by FSDP
            **kwargs,
        )
        self.cluster.scale(parallel_mode_config.num_workers)
        self._initialize_workers(config, parallel_mode_config)

    def _init_slurm_cluster(
        self, config: TransformerHandlerConfig, parallel_mode_config: ParallelModeConfig
    ):
        """Initialize a SLURM-based Dask cluster with GPU allocation.

        Note: Dask's integration with SLURM currently only supports allocating single entire node
        at a time, with each node running as a single SLURM task. This implementation adapts
        to that limitation by requesting complete nodes and running multiple workers (one per GPU)
        within each node. If our cluster eventually supports GRES (Generic Resource) scheduling,
        this implementation could be modified to allow for more granular GPU allocation across
        nodes rather than requiring full node allocation (I think, needs to be tested).
        """
        # Lazy import because dask_jobqueue cannot be started in a subprocess, which
        # happens e.g. with streamlit
        from dask_jobqueue.slurm import SLURMCluster

        # Validate that num_workers is divisible by num_gpus_per_node
        num_gpus_per_node = parallel_mode_config.num_gpus_per_node
        if parallel_mode_config.num_workers % num_gpus_per_node != 0:
            raise ValueError(
                f"Number of workers ({parallel_mode_config.num_workers}) must be divisible by "
                f"num_gpus_per_node ({num_gpus_per_node}). We assume each node has {num_gpus_per_node} GPUs, "
                f"and current dask-jobqueue infrastructure only supports allocating whole nodes. "
            )
            # TODO: add support for gres when available in our cluster for partial node allocation

        # Calculate number of jobs needed (each job = 1 slurm node with num_gpus_per_node GPUs)
        num_jobs = parallel_mode_config.num_workers // num_gpus_per_node

        log_dir = parallel_mode_config.log_directory
        os.makedirs(log_dir, exist_ok=True)

        # Calculate total memory needed per node (memory_per_worker * num_gpus_per_node)
        memory_per_worker = parallel_mode_config.memory_per_worker
        MEMORY_UNIT_LENGTH = 2  # Memory units are typically 2 chars (e.g. "GB", "MB")
        value = int(
            memory_per_worker[:-MEMORY_UNIT_LENGTH]
        )  # Get numeric value by removing last 2 chars (e.g. "GB")
        unit = memory_per_worker[-MEMORY_UNIT_LENGTH:]  # Get unit (e.g. "GB")
        assert len(unit) == MEMORY_UNIT_LENGTH, (
            f"Memory unit must be {MEMORY_UNIT_LENGTH} characters long, got {unit}"
        )
        total_memory = f"{value * parallel_mode_config.num_gpus_per_node}{unit}"

        self.cluster = SLURMCluster(
            cores=parallel_mode_config.num_cpus_per_worker * num_gpus_per_node,
            memory=total_memory,
            processes=num_gpus_per_node,  # Each job runs num_gpus_per_node dask workers (one per GPU)
            walltime=parallel_mode_config.walltime,
            job_extra_directives=[
                "--nodes=1",  # Always request 1 node per job
                "--exclusive",  # Exclusive node access
                "--mem=0",  # Use all available memory
                f"--cpus-per-task={parallel_mode_config.num_cpus_per_worker}",
                f"-o {log_dir}/job_%j_task_%t.out",
                f"-e {log_dir}/job_%j_task_%t.err",
            ],
            log_directory=log_dir,
        )

        # Scale jobs to the required number of jobs
        self.cluster.scale(jobs=num_jobs)
        self._initialize_workers(config, parallel_mode_config)

    def _initialize_workers(
        self, config: TransformerHandlerConfig, parallel_mode_config: ParallelModeConfig
    ):
        self.client = Client(self.cluster)
        self.client.wait_for_workers(parallel_mode_config.num_workers)

        def get_cuda_visible_devices() -> int | None:
            device = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if device is not None:
                # If has several devices, assume the first one is the one to use for that worker
                if "," in device:
                    device = device.split(",", maxsplit=1)[0]
                    os.environ["CUDA_VISIBLE_DEVICES"] = device
                return int(device)
            return None

        worker_to_cuda_device = self.client.run(get_cuda_visible_devices)
        workers_info = self.client.scheduler_info()["workers"]
        sorted_workers = dict(
            sorted(workers_info.items(), key=lambda item: item[1]["id"])
        )
        # The first worker is the master in the torch distributed setup
        master_addr = next(iter(sorted_workers.values()))["host"]

        futures = []
        worker_ids = []
        for rank, (worker_address, worker_data) in enumerate(sorted_workers.items()):
            worker_id = worker_data["id"]
            # On some occasions, dask SLURM integration auto assigns CUDA_VISIBLE_DEVICES, otherwise we set it here
            worker_cuda_device = worker_to_cuda_device[worker_address]
            if worker_cuda_device is None:
                worker_cuda_device = rank % parallel_mode_config.num_gpus_per_node

            parallel_worker_config = ParallelWorkerConfig(
                rank=rank,
                world_size=parallel_mode_config.num_workers,
                local_rank=worker_cuda_device,
                master_addr=master_addr,
                master_port=parallel_mode_config.torch_port,
                **parallel_mode_config.model_dump(),
            )
            future_op = self.client.submit(
                ParallelTransformerHandler,
                config=config,
                parallel_worker_config=parallel_worker_config,
                workers=[worker_id],
                actor=True,
            )
            futures.append(future_op)
            worker_ids.append(worker_id)

        self.actors: list[Actor] = self._client_gather(futures)
        self.worker_ids = worker_ids

    async def __call__(
        self,
        inputs: str | BatchEncoding | list[dict] | None = None,
        tools_json: list[dict] | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[str, torch.Tensor]:
        if inputs is None:
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        inputs_tokenized = _get_tokenized_inputs(self.tokenizer, inputs, tools_json)
        inputs_len = inputs_tokenized["input_ids"].shape[1]
        outputs = await AsyncBufferedWorker.__call__(self, **inputs_tokenized, **kwargs)
        AsyncTransformer._maybe_finalize_logits_processors(
            kwargs.get("logits_processor"), outputs
        )

        return _process_outputs(self.config, self.tokenizer, outputs, inputs_len)

    async def _batched_call(self, batch_kwargs: dict[str, Any]):
        return self._submit_and_gather(
            self.handler_call_fn, **batch_kwargs, split_data=True
        )

    def _submit_and_gather(
        self,
        func: Callable[Concatenate[ParallelTransformerHandler, TParams], TReturn] | str,
        *args,
        split_data: bool = False,
        **kwargs,
    ) -> list[TReturn]:
        """Submit a function to all workers and gather the results.

        Args:
            func: The function to send to each worker. If a string is provided,
                then getattr(handler, func) is used. If func is not a string,
                the first argument must be the ParallelTransformerHandler that it will
                be executed on.
            split_data: If True, split the data between workers. If False,
                send the same data to all workers.
            *args: Positional arguments to pass to the method.
            **kwargs: Keyword arguments to pass to the method.

        Returns:
            The gathered results from the workers.
        """
        if split_data:
            chunker = TensorChunker(
                num_chunks=len(self.actors),
            )
            split_args, split_kwargs, dummy_flags = chunker.chunkify(*args, **kwargs)
        else:
            split_args = [args] * len(self.actors)
            split_kwargs = [kwargs] * len(self.actors)

        futures = [
            handler._exec_func(
                func,
                *args_i,
                **kwargs_i,
            )
            for handler, worker_id, args_i, kwargs_i in zip(
                self.actors, self.worker_ids, split_args, split_kwargs, strict=True
            )
        ]
        results: list[TReturn] = self._client_gather(futures)

        if split_data:
            return chunker.dechunkify(results, dummy_flags)
        return results

    def wrap_afunc(
        self,
        func: Callable[
            Concatenate[ParallelTransformerHandler, TParams], Awaitable[TReturn]
        ],
        **kwargs,
    ) -> Callable[TParams, Awaitable[TReturn]]:
        raise NotImplementedError(
            "ParallelAsyncTransformer does not implement wrap_afunc(). "
            "Wrap a synchronous function with wrap_func() instead."
        )

    @overload
    def wrap_func(
        self,
        *,
        worker_agg_fn: Callable[[list[TReturn]], TReturn] | None = None,
        **kwargs,
    ) -> Callable[
        [Callable[Concatenate[ParallelTransformerHandler, TParams], TReturn]],
        Callable[TParams, TReturn],
    ]: ...

    @overload
    def wrap_func(
        self,
        func: Callable[Concatenate[ParallelTransformerHandler, TParams], TReturn],
        *,
        worker_agg_fn: Callable[[list[TReturn]], TReturn] | None = None,
        **kwargs,
    ) -> Callable[TParams, TReturn]: ...

    def wrap_func(
        self,
        func: (
            Callable[Concatenate[ParallelTransformerHandler, TParams], TReturn] | None
        ) = None,
        *,
        worker_agg_fn: Callable[[list[TReturn]], TReturn] | None = None,
        **kwargs,
    ) -> Callable:
        """Wrap a function to execute on all workers and return gathered results.

        Args:
            func: The function to wrap.
            worker_agg_fn: A function to aggregate the results from all workers.
            kwargs: Arguments that are discarded. Included here to enable a
                subclass to add additional arguments.
        """
        if worker_agg_fn is None:
            raise ValueError("worker_agg_fn must be provided.")

        if func is None:
            return partial(self.wrap_func, worker_agg_fn=worker_agg_fn, **kwargs)

        @wraps(func)
        def wrapped_func(*args, **kwargs) -> TReturn:
            return worker_agg_fn(self._submit_and_gather(func, *args, **kwargs))

        return wrapped_func

    def state_dict(self, **kwargs) -> dict[str, torch.Tensor]:
        """Get consolidated state dict from all workers.

        With FSDP2, we need to manually consolidate the state dict
        """

        def state_dict_worker(
            handler: ParallelTransformerHandler,
        ) -> dict[str, torch.Tensor]:
            state_dict = handler.module.state_dict()
            # Convert DTensors to full tensors
            for key, tensor in state_dict.items():
                if hasattr(tensor, "full_tensor"):
                    state_dict[key] = tensor.full_tensor()
            return state_dict

        # Only need the state dict from rank 0
        state_dict = self._submit_and_gather(state_dict_worker, **kwargs)[0]
        return {k: v.cpu() for k, v in state_dict.items()}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        # For some reason, Dask hangs when we pass a large object (e.g. state_dict)
        # directly to the workers. I can replicate it with the following:
        #
        # @handler.wrap_func
        # def hello(handler, _):
        #     print("hello")
        #
        # hello([0] *  1_000_000)
        # NOTE: this does not seem to be FSDP-related, as the issue didn't go away when
        # I disabled FSDP.
        raise NotImplementedError(
            "ParallelAsyncTransformer.load_state_dict() is not implemented yet. It is"
            " recommended to use .save_checkpoint() and .load_checkpoint() instead. "
        )

    def load_checkpoint(self, ckpt: os.PathLike | str) -> None:
        self._submit_and_gather("load_checkpoint", ckpt)

    def save_checkpoint(self, ckpt: os.PathLike | str, **kwargs) -> None:
        self._submit_and_gather("save_checkpoint", ckpt, **kwargs)

    def teardown(self) -> None:
        if self._initialized:
            self.client.shutdown()
            self.cluster.close()
            del self.client
            del self.cluster
            self._initialized = False

    def __del__(self) -> None:
        self.teardown()

    @staticmethod
    def _wrap_dask_future(dask_future: ActorFuture):
        """Converts a Dask ActorFuture into an awaitable asyncio.Future."""
        loop = asyncio.get_running_loop()
        return asyncio.ensure_future(loop.run_in_executor(None, dask_future.result))

    @staticmethod
    def _raise_exceptions(done, pending, wrapped_futures):
        exceptions = []
        for future in done:
            exc = future.exception()
            if exc:
                exceptions.append(exc)
        if exceptions:
            if len(exceptions) == 1:
                raise exceptions[0]
            raise ExceptionGroup("Multiple actor exceptions", exceptions)

        if pending:
            pending_indices = sorted([wrapped_futures.index(p) for p in pending])
            raise TimeoutError(
                f"Tasks didn't complete within timeout. {len(pending)} out of {len(wrapped_futures)} "
                f"still pending. Pending task indices: {pending_indices}"
            )

    async def _client_gather_async(self, futures):
        """Gather results from futures, propagating exceptions as they arrive.

        Unlike client.gather() which waits for all futures to complete before raising
        any exceptions, this method processes futures as they complete and raises
        exceptions immediately. This is crucial when using FSDP where workers may
        be stuck waiting for each other when one worker crashes, causing long hangs.

        Note: Dask Actors currently have an issue where they're not working properly with
        dask.gather() and can cause blocking issues or hide worker errors. This implementation
        works around those limitations.
        """
        try:
            wrapped_futures = [self._wrap_dask_future(f) for f in futures]

            # Use asyncio.wait with FIRST_EXCEPTION instead of gather
            done, pending = await asyncio.wait(
                wrapped_futures, timeout=1200, return_when=asyncio.FIRST_EXCEPTION
            )

            self._raise_exceptions(done, pending, wrapped_futures)

            return await asyncio.gather(*wrapped_futures)
        except Exception:
            logger.exception("Error in dask workers: %s")
            for future in wrapped_futures:
                future.cancel()
            self.teardown()
            # sys.exit(1) would wait for dask to finish, which can cause hanging
            # when workers are in a deadlock. Use os._exit to force immediate termination
            # TODO: this is more of a hack, we should propagate special exception that is
            # not caught by the rollout manager.
            os._exit(1)

    def _client_gather(self, futures: list[ActorFuture]) -> list[Any]:
        # Use distributed.utils.sync to run the async function in the current thread
        return sync(self.client.loop, self._client_gather_async, futures)  # type: ignore[arg-type]
