from __future__ import annotations

import asyncio
import atexit
import logging
import os
import socket
import sys
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from enum import StrEnum, auto
from functools import cache, partial, wraps
from pathlib import Path
from typing import (
    Any,
    Concatenate,
    Literal,
    ParamSpec,
    Self,
    TypeVar,
    assert_never,
    cast,
)

import accelerate
import torch
import torch.distributed as dist
import tree
from dask import config
from dask.distributed import Actor, ActorFuture, Client
from distributed.utils import sync
from pydantic import BaseModel, ConfigDict, Field, field_validator
from torch import nn
from torch.cuda import nccl
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullStateDictConfig,
    FullyShardedDataParallel,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.nn.functional import pad
from transformers import GenerationMixin, PreTrainedModel, PreTrainedTokenizer
from transformers.generation.utils import GenerateDecoderOnlyOutput, LogitsProcessorList
from transformers.tokenization_utils_base import BatchEncoding

from ldp.graph.async_torch import AsyncBufferedWorker, AsyncTorchModule
from ldp.nn.generation import LogitsProcessorWithFinalize
from ldp.nn.handlers.chunking import TensorChunker, TOutputType
from ldp.nn.handlers.module_handler import ModuleExecutionInterface, ModuleHandler
from ldp.nn.lm_config import LMConfig, TorchDType
from ldp.nn.utils import REPO_ROOT, set_seed

if sys.version_info >= (3, 12):
    from typing import overload
else:
    from typing_extensions import overload  # noqa: UP035

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


def is_conversation(messages) -> bool:
    """Check if messages is an instance of Conversation."""
    return isinstance(messages, list) and all(
        isinstance(msg, dict)
        and all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in msg.items()
        )
        for msg in messages
    )


def get_unused_port() -> int:
    """Find an unused port by creating a temporary socket."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class ExecutionMode(StrEnum):
    LOCAL_MACHINE = auto()
    SLURM_CLUSTER = auto()


class FSDPConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    offload_cpu: bool = Field(
        default=False,
        description="Whether to offload model parameters and gradients to CPU.",
    )
    activation_checkpointing: bool = False
    cpu_ram_efficient_loading: bool = False
    state_dict_type: StateDictType = StateDictType.FULL_STATE_DICT
    backward_prefetch: BackwardPrefetch | None = Field(
        BackwardPrefetch.BACKWARD_PRE,
        description=(
            "See https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.BackwardPrefetchDefault"
            " is PRE to be consistent with FSDP's default."
        ),
    )

    @field_validator("backward_prefetch", mode="before")
    @classmethod
    def validate_backward_prefetch(cls, v) -> BackwardPrefetch | None:
        if v is None:
            return None
        if isinstance(v, str):
            # BackwardPrefetch isn't a StrEnum, so Pydantic won't cast a string
            return BackwardPrefetch[v]
        return BackwardPrefetch(v)


class ParallelModeConfig(FSDPConfig):
    num_workers: int = Field(description="Number of workers to be used in the cluster.")
    num_cpus_per_worker: int = Field(
        default=1, description="Number of CPUs to be allocated per worker."
    )
    execution_mode: ExecutionMode = Field(
        description=(
            "Execution mode of the current setup, defines how to allocate resources "
            "(cpus/gpus) for the model to run on."
        ),
        default=ExecutionMode.LOCAL_MACHINE,
    )

    scheduler_addr: str = "localhost"
    scheduler_port: int = Field(default=0, description="0 means Dask picks randomly.")
    torch_port: int = Field(default_factory=get_unused_port)

    # Below configurations (walltime, memory, log_directory) are only for ExecutionMode.SLURM_CLUSTER:
    walltime: str = Field(
        default="00:30:00", description="Max time the worker can run."
    )
    memory: str = Field(default="32GB", description="Memory allocated per worker.")
    log_directory: str = Field(
        default=f"{REPO_ROOT}/logs/slurm_outputs/",
        description="Directory to store logs.",
    )


class LMType(StrEnum):
    GENERATION = auto()
    REGRESSION = auto()


class TransformerHandlerConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    lm_config: LMConfig
    lm_type: LMType
    checkpoint: str | None = None

    batch_size: int
    max_wait_interval: float = 0.1
    module_call_fn: Callable
    collate_fn: Callable
    decollate_fn: Callable

    parallel_mode_config: ParallelModeConfig | None = Field(
        default=None,
        description=(
            "Optional configuration for distributing the transformer across "
            "multiple devices/nodes. If not provided, will default to single-device."
        ),
    )

    def make_async_module(self, **kwargs) -> AsyncTransformerInterface:
        if self.parallel_mode_config:
            return ParallelAsyncTransformer(config=self, **kwargs)
        return AsyncTransformer(config=self, **kwargs)


class AsyncTransformerInterface(ModuleExecutionInterface, AsyncTorchModule, ABC):
    """Base class for async interactions with a transformer model."""

    @abstractmethod
    async def __call__(  # type: ignore[override]
        self,
        inputs: str | BatchEncoding | list[dict],
        tools_json: list[dict[Any, Any]] | None = None,
        **kwargs,
    ) -> tuple[str, torch.Tensor]:
        """Call the transformer on a single input, which may be encoded."""

    @staticmethod
    def model_generate(model: PreTrainedModel, *args, **kwargs):
        """A method that can be used as module_call_fn to sample from an LLM."""
        # Summoning params per https://github.com/pytorch/pytorch/issues/100069
        # If model is not FSDP, this context manager is a no-op.
        with FullyShardedDataParallel.summon_full_params(model, recurse=False):
            logger.debug(
                f"model.generate() input_ids shape: {kwargs['input_ids'].shape}, rank"
                f" {os.environ.get('RANK')}"
            )
            if not isinstance(model, GenerationMixin):  # type: ignore[unreachable]
                raise TypeError(
                    "model_generate only supports models that inherit from GenerationMixin"
                )
            return model.generate(  # type: ignore[unreachable]
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

        self._setup_accelerator()

        if config.checkpoint is not None:
            self.load_checkpoint(config.checkpoint)

    def _setup_accelerator(self):
        self.accelerator = accelerate.Accelerator(
            # This has to be disabled because accelerator wraps forward() to upcast outputs to fp32. That
            # causes problems with generation, where the cache is expected to be in the same dtype as the model.
            # TODO: understand why this doesn't break in huggingface code and re-enable in ours.
            # mixed_precision=("bf16" if self.config.lm_config.dtype == TorchDType.bf16 else "no")
        )
        self.module = self.accelerator.prepare(self.module)

    @property
    def local_rank(self) -> int:
        return int(os.getenv("LOCAL_RANK", "0"))

    def load_checkpoint(self, ckpt: os.PathLike | str, **kwargs) -> None:
        logger.info(f'Loading checkpoint from "{ckpt}"')
        self.accelerator.load_state(str(ckpt), **kwargs)

    def save_checkpoint(self, ckpt: os.PathLike | str, **kwargs) -> None:
        self.accelerator.save_state(str(ckpt), **kwargs)
        self.barrier()
        # We do not want to save random states - they would be loaded by load_state
        # automatically. Clean up after all processes have saved.
        if int(os.getenv("RANK", "0")) == 0:
            random_states = Path(ckpt).rglob("random_state*.pkl")
            for random_state in random_states:
                random_state.unlink()

    @staticmethod
    def barrier() -> None:
        if dist.is_initialized():
            dist.barrier()


class AsyncTransformer(TransformerHandler, AsyncTransformerInterface):
    """Equivalent of AsyncTorchModule, but for a transformer model.

    The model is instantiated in the current process.
    """

    def __init__(self, config: TransformerHandlerConfig):
        # First init the local module
        TransformerHandler.__init__(self, config)
        # Then init the logic to buffer inputs to the module
        AsyncTorchModule.__init__(
            self,
            module=self.module,
            batch_size=config.batch_size,
            max_wait_interval=config.max_wait_interval,
            collate_fn=config.collate_fn,
            decollate_fn=config.decollate_fn,
            module_call_fn=config.module_call_fn,
        )

    async def __call__(
        self,
        inputs: str | BatchEncoding | dict | list[dict] | None = None,
        tools_json: list[dict[Any, Any]] | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[str, torch.Tensor]:
        if inputs is None:
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        inputs_tokenized = _get_tokenized_inputs(self.tokenizer, inputs, tools_json).to(
            device=_get_data_device()
        )
        inputs_len = inputs_tokenized["input_ids"].shape[1]
        self.module.eval()
        outputs = await AsyncTorchModule.__call__(self, **inputs_tokenized, **kwargs)
        self._maybe_finalize_logits_processors(kwargs.get("logits_processor"), outputs)

        return _process_outputs(self.config, self.tokenizer, outputs, inputs_len)

    @staticmethod
    def _maybe_finalize_logits_processors(
        logits_processors: LogitsProcessorList | None,
        outputs: GenerateDecoderOnlyOutput,
    ) -> None:
        if not logits_processors:
            return
        for processor in logits_processors:
            if isinstance(processor, LogitsProcessorWithFinalize):
                processor.finalize(outputs.sequences)


class ParallelWorkerConfig(FSDPConfig):
    model_config = ConfigDict(extra="ignore")

    rank: int = Field(description="Rank of the current process.")
    world_size: int = Field(description="Total number of processes.")
    local_rank: int = Field(description="Local rank of the current process.")
    master_addr: str = Field(description="Address of the master node.")
    master_port: int = Field(description="Port of the master node.")

    def set_env_vars(self):
        # These inform torch.distributed how to set up the process group
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.local_rank)
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)

        # These tell PretrainedModel.from_pretrained() that we're using FSDP
        os.environ["ACCELERATE_USE_FSDP"] = "1"
        os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = str(
            int(self.cpu_ram_efficient_loading)
        )


class ParallelTransformerHandler(TransformerHandler):
    def __init__(
        self,
        config: TransformerHandlerConfig,
        parallel_worker_config: ParallelWorkerConfig,
    ):
        parallel_worker_config.set_env_vars()
        dist.init_process_group(backend="nccl")
        self.worker_config = parallel_worker_config
        super().__init__(config)

    def _setup_accelerator(self):
        bf16 = self.config.lm_config.dtype == TorchDType.bf16

        mixed_precision = None
        if bf16:
            bf16_ready = (
                torch.version.cuda
                and torch.cuda.is_bf16_supported()
                and torch.version.cuda >= "11.0"
                and dist.is_nccl_available()
                and nccl.version() >= (2, 10)
            )
            assert bf16_ready, (
                "Mixed precision training requires CUDA 11.0 and NCCL 2.10"
            )
            mixed_precision = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )

        self.accelerator = accelerate.Accelerator(
            # See note in TransformerHandler._setup_accelerator() about this
            # mixed_precision=("bf16" if bf16 else "no"),
            fsdp_plugin=accelerate.FullyShardedDataParallelPlugin(
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision_policy=mixed_precision,
                auto_wrap_policy="transformer_based_wrap",
                cpu_offload=self.worker_config.offload_cpu,
                activation_checkpointing=self.worker_config.activation_checkpointing,
                cpu_ram_efficient_loading=self.worker_config.cpu_ram_efficient_loading,
                sync_module_states=self.worker_config.cpu_ram_efficient_loading,
                state_dict_type=self.worker_config.state_dict_type,
                backward_prefetch=self.worker_config.backward_prefetch,
            ),
        )

        if self.config.lm_config.device == "meta":
            self.module = prepare_model_for_fsdp_with_meta_device(self.module)

        self.module = self.accelerator.prepare(self.module)

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
            device_type: str = str(self.module.device.type)
            assert self.module.dtype is not None
            dtype: torch.dtype = self.module.dtype  # type: ignore[assignment]
            with torch.autocast(device_type=device_type, dtype=dtype):
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

        # This uses NVIDIA's NVML layer instead of native CUDA, which is more robust in GPU detection
        # post initialization. This prevents issues with forked processes wrongly detecting the
        # default GPU as cuda:0
        os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
        self.cluster = LocalCUDACluster(
            n_workers=parallel_mode_config.num_workers,
            threads_per_worker=parallel_mode_config.num_cpus_per_worker,
            host=parallel_mode_config.scheduler_addr,
            port=parallel_mode_config.scheduler_port,
            memory_limit=None,  # do not let Dask manage memory - if we OOM, we OOM
        )
        self._initialize_workers(config, parallel_mode_config)

    def _init_slurm_cluster(
        self, config: TransformerHandlerConfig, parallel_mode_config: ParallelModeConfig
    ):
        """Initialize a SLURM-based Dask cluster with GPU allocation."""
        # Lazy import because dask_jobqueue cannot be started in a subprocess, which
        # happens e.g. with streamlit
        from dask_jobqueue import SLURMCluster

        self.cluster = SLURMCluster(
            cores=parallel_mode_config.num_cpus_per_worker,
            memory=parallel_mode_config.memory,
            processes=1,  # Single dask worker per slurm worker
            walltime=parallel_mode_config.walltime,
            job_extra=[
                "--gres=gpu:1"
            ],  # 1 GPU per worker seems to be the common case for now
            log_directory=parallel_mode_config.log_directory,
        )
        self._initialize_workers(config, parallel_mode_config)

    def _initialize_workers(
        self, config: TransformerHandlerConfig, parallel_mode_config: ParallelModeConfig
    ):
        self.cluster.scale(parallel_mode_config.num_workers)
        self.client = Client(self.cluster)
        self.client.wait_for_workers(parallel_mode_config.num_workers)

        def get_cuda_visible_devices() -> int | None:
            device = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if device is not None:
                # If has several devices, assume the first one is the one to use for that worker
                if "," in device:
                    device = device.split(",", maxsplit=1)[0]
                    os.environ["CUDA_VISIBLE_DEVICES"] = device
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
            worker_cuda_device = worker_to_cuda_device[worker_address]
            if worker_cuda_device is None:
                assert (
                    parallel_mode_config.execution_mode != ExecutionMode.SLURM_CLUSTER
                ), "CUDA_VISIBLE_DEVICES should be pre set for SLURM workers."
                worker_cuda_device = rank

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
        inputs: str | BatchEncoding | dict | list[dict] | None = None,
        tools_json: list[dict[Any, Any]] | None = None,
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

    @overload
    def _submit_and_gather(
        self,
        func: Callable[Concatenate[ParallelTransformerHandler, TParams], TOutputType]
        | str,
        *args,
        split_data: Literal[True],
        **kwargs,
    ) -> TOutputType: ...

    @overload
    def _submit_and_gather(
        self,
        func: Callable[Concatenate[ParallelTransformerHandler, TParams], TReturn] | str,
        *args,
        split_data: Literal[False] = False,  # default
        **kwargs,
    ) -> list[TReturn]: ...

    def _submit_and_gather(
        self,
        func: Callable[Concatenate[ParallelTransformerHandler, TParams], Any] | str,
        *args,
        split_data: bool = False,
        **kwargs,
    ) -> TOutputType | list[TReturn]:
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
        results = self._client_gather(futures)

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
        # do this manually - accelerator.get_state_dict doesn't expose all options
        def state_dict_worker(
            handler: ParallelTransformerHandler, **kwargs
        ) -> dict[str, torch.Tensor]:
            # For an explanation, see:
            # https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullStateDictConfig
            cfg = FullStateDictConfig(
                **({"offload_to_cpu": True, "rank0_only": True} | kwargs)
            )
            with FullyShardedDataParallel.state_dict_type(
                handler.module, StateDictType.FULL_STATE_DICT, cfg
            ):
                # will be the full state on rank 0 and empty on the others
                return handler.module.state_dict()

        # Only the 0th rank returns the full state dict
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


# Helpers


# TODO: We should use the tokenizer to manage this instead of calling pad ourselves.
def _collate_fn_transformer(
    samples: list[dict[str, torch.Tensor]],
    pad_token_id: int,
    left_pad: bool,
    agg_keys: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Collates and pads a batch of samples for input into a huggingface transformer model."""
    if agg_keys is None:
        agg_keys = {"input_ids", "attention_mask"}
    seq_lens = [inp["input_ids"].shape[1] for inp in samples]
    max_seq_len = max(seq_lens)
    n_pads = [max_seq_len - seq_len for seq_len in seq_lens]

    # TODO: these pad_value and pad side lookups might be getting expensive. Refactor
    pad_value = {
        "input_ids": pad_token_id,
        "attention_mask": 0,
    }

    batch = {
        key: torch.cat(
            [
                # see comment in TransformerHandler.__init__ about why we're left padding.
                pad(
                    inp[key],
                    ((n_pad, 0) if left_pad else (0, n_pad)),
                    value=pad_value.get(key, 0),
                )
                for inp, n_pad in zip(samples, n_pads, strict=True)
            ],
            dim=0,
        )
        for key in agg_keys
    }

    # Treating other keys as constant kwargs params for the model
    other_keys = set(samples[0].keys()) - agg_keys
    for key in other_keys:
        for sample in samples:
            if key not in sample:
                raise ValueError(f"Missing key {key} in sample.")
            if key in batch and batch[key] != sample[key]:
                raise ValueError(
                    f"Constant kwarg key {key} has different values within batch."
                )
            batch[key] = sample[key]

    return batch


# be explicit
collate_fn_transformer_left_pad = partial(_collate_fn_transformer, left_pad=True)
collate_fn_transformer_right_pad = partial(_collate_fn_transformer, left_pad=False)


def decollate_fn_transformer_decoder(
    batched_output: GenerateDecoderOnlyOutput,
) -> list[GenerateDecoderOnlyOutput]:
    """Decollates a batched output from a huggingface transformer decoder."""
    batch_size = len(batched_output.sequences)
    outputs: list[GenerateDecoderOnlyOutput] = [
        GenerateDecoderOnlyOutput(
            sequences=batched_output.sequences[i][None, :],  # type: ignore[arg-type]
            scores=[
                score[i][None, :]
                for score in batched_output.scores
                if (score[i] is not None)
            ]
            if batched_output.scores
            else None,  # type: ignore[arg-type]
        )
        for i in range(batch_size)
    ]
    # There are other fields in the batched output that we can add here,
    # but no calling code is using them, so ignore for now.
    return outputs


# From: https://github.com/pytorch/torchtune/blob/c5db813ce0473db090a4f1f6b450f559acac58e5/torchtune/training/_distributed.py#L207
# NOTE: I removed the LoRA logic from the source code since we don't use torchtune for now.
# See issue in docstring for why we needed this.
def prepare_model_for_fsdp_with_meta_device(model: nn.Module) -> nn.Module:
    """Dynamically define reset_parameters on every submodule of the model.

    More details here: https://github.com/pytorch/pytorch/issues/104187.

    Args:
        model: model to prepare for usage with FSDP and meta device.

    Returns:
        Model with reset_parameters defined on every submodule.

    Raises:
        RuntimeError: if model contains submodule with non-callable attribute reset_parameters
    """
    for k, v in model.named_modules():
        # If the module does not have reset_parameters defined, we define
        # a no-op reset_parameters method to satisfy FSDP's contract.
        reset_params = getattr(v, "reset_parameters", None)

        if reset_params is not None and not callable(reset_params):
            raise RuntimeError(
                "Cannot override existing reset_parameters variable for FSDP init"
                f" in {k}"
            )

        if reset_params is None:
            v.reset_parameters = _dummy_reset_params.__get__(v)  # pylint: disable=assignment-from-no-return

    return model


def _dummy_reset_params(_: nn.Module) -> None:
    """Dummy method for patching no-op reset_parameters() when using FSDP with meta device."""


@cache
def _get_data_device() -> torch.device:
    """Get the device where input tensors should be placed."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _get_tokenized_inputs(
    tokenizer: PreTrainedTokenizer,
    inputs: str | dict | BatchEncoding | list[dict],
    tools_json: list[dict[Any, Any]] | None = None,
) -> BatchEncoding:
    if isinstance(inputs, BatchEncoding):
        return inputs
    if isinstance(inputs, dict):
        return BatchEncoding(inputs)
    if isinstance(inputs, str):
        return tokenizer(inputs, return_tensors="pt")
    if is_conversation(inputs):
        result = tokenizer.apply_chat_template(
            inputs,
            tools=tools_json,  # type: ignore[arg-type]
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
        return cast(BatchEncoding, result)
    raise ValueError(
        f"inputs must be a str, BatchEncoding, or Conversation, but got {type(inputs)}"
    )


def _process_outputs(
    config: TransformerHandlerConfig,
    tokenizer: PreTrainedTokenizer,
    outputs,
    input_len: int,
) -> tuple[str, torch.Tensor]:
    outputs = tree.map_structure(lambda x: x.cpu(), outputs)

    if config.lm_type == LMType.REGRESSION:
        # TODO: deal with different output types w @overload
        # dask cannot serialize precisions lower than fp32
        return "", outputs.float()

    assert len(outputs.sequences) == 1, (
        "Expected a batch size of 1, as AsyncBufferedWorker handles batching internally."
    )
    output_sequence = outputs.sequences[0]

    if outputs.scores:
        output_sequence = output_sequence[-len(outputs.scores) :]
    else:
        # This way of finding the output sequence can be messed up by tokenizers that
        # have overlapping eos/bos/pad tokens. We should consider removing it and relying
        # on `outputs.scores` exclusively.
        def _find_first_non_padding_index(
            sequence: torch.Tensor, pad_token_id: int
        ) -> int:
            """Find the first non-padding index in a sequence."""
            padding_mask = (sequence == pad_token_id).long()
            cumsum_padding = padding_mask.cumsum(dim=0)  # noqa: FURB184
            sequence_length = sequence.size(0)
            expected_cumsum = torch.arange(1, sequence_length + 1)
            divergence = (cumsum_padding != expected_cumsum).long()
            # Find the first index where the cumsums diverge. This is the first non-padding index.
            return int(divergence.argmax(dim=0).item())

        padding_len = _find_first_non_padding_index(
            output_sequence, tokenizer.pad_token_id
        )
        # Take the output sequence starting after the initial padding and input length
        output_sequence = output_sequence[padding_len + input_len :]

    out_text = tokenizer.decode(output_sequence, skip_special_tokens=True)

    # Extract logprobs
    if outputs.scores:
        logits = torch.cat(outputs.scores, dim=0).to(torch.float32)
        if len(logits) != len(output_sequence):
            discarded_logits = logits[len(output_sequence) :]
            assert torch.all(discarded_logits == -float("inf")), (
                "Discarded logits should be invalid ones."
            )
            # Discard the logits for post-eos tokens
            logits = logits[: len(output_sequence)]
        logprobs = logits_to_logprobs(logits, output_sequence)
    else:
        logprobs = torch.tensor([])

    return out_text, logprobs


def logits_to_logprobs(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Convert logits to log probabilities.

    Roughly following https://huggingface.co/Pwicke/logprobs_for_CausalLMs

    Note that PretrainedModel.generate returns a tuple of logits, but
    TransformerHandler._process_outputs converts it into tensor for us.
    """
    logprobs = nn.functional.log_softmax(logits, dim=-1)
    # Can also be written as logprobs[torch.arange(len(token_ids)), token_ids]
    # but gather is more efficient.
    return torch.gather(logprobs, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)


def _move_tensor(x, device: torch.device) -> torch.Tensor:
    return x.to(device) if isinstance(x, torch.Tensor) else x


def maybe_set_tokenizer_chat_template(
    tokenizer: PreTrainedTokenizer, chat_template: str | None
) -> None:
    """Set the chat template for the tokenizer if needed."""
    # Check if the tokenizer's existing chat_template contains the '{% generation %}' tag

    model_name = tokenizer.name_or_path.lower()
    if in_test := ("PYTEST_CURRENT_TEST" in os.environ and "gpt2" in model_name):
        # Override whatever was passed in for tests
        logger.info("Using llama3 chat template for gpt2 model in tests.")
        chat_template = "llama3_chat_template_ori.jinja"

    if chat_template is not None:
        # User should provided a chat template that includes the {% generation %} keyword needed
        # for populating "assistant_masks" when tokenizing inputs for the purpose of training.
        # We need to monitor the PR comments at https://github.com/huggingface/transformers/pull/30650
        # for updates on the issue.
        template_path = Path(chat_template)
        if not template_path.exists():
            # Check the aviary_internal library of templates
            template_path = Path(f"{REPO_ROOT}/ldp/nn/chat_templates/{chat_template}")
        loaded_chat_template = template_path.read_text()

        if in_test:
            # Adapt the template for the actual tokenizer we are using
            loaded_chat_template = loaded_chat_template.replace(
                "<|eot_id|>", tokenizer.eos_token
            )
        tokenizer.chat_template = loaded_chat_template
        return

    if not tokenizer.chat_template or "{% generation %}" in tokenizer.chat_template:
        # Warn the user about potential training issues
        logger.warning(
            "Tokenizer does not have a chat template with '{% generation %}'."
            " Generative training will have issues as the tag does not exist, which"
            " means HuggingFace's internal code for retrieving the assistant_mask for"
            " training will be invalid. Fine-tuning for other purposes, like"
            " regression, will not be affected."
        )
