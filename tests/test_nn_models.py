import asyncio
from functools import partial
from itertools import starmap
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp

import pytest
import torch
from torch.distributed.fsdp import StateDictType
from torch.optim import SGD
from transformers.generation.utils import GenerateDecoderOnlyOutput

import ldp.nn
from ldp.nn.handlers.transformer_handler import get_unused_port

from .conftest import PARALLEL_MODE_CONFIGS, TEST_GPUS


def test_model_load():
    config = ldp.nn.LMConfig(model="gpt2")
    config.resolve_model_location()
    config.get_causal_lm()


class TestTensorChunker:
    def test_chunkify_add_dummy_chunks(self):
        batch_size = 3
        num_chunks = 5

        sample_tensor = torch.arange(1, batch_size * 10 + 1).reshape(batch_size, 10)

        chunker = ldp.nn.TensorChunker(num_chunks=num_chunks)
        split_args, split_kwargs, dummy_chunk_flags = chunker.chunkify(sample_tensor)

        assert len(split_args) == num_chunks
        assert len(split_kwargs) == num_chunks
        assert dummy_chunk_flags == [False, False, False, True, True]
        assert torch.equal(split_args[0][0], sample_tensor[:1])
        assert torch.equal(split_args[1][0], sample_tensor[1:2])
        assert torch.equal(split_args[2][0], sample_tensor[2:3])
        assert torch.equal(split_args[3][0], sample_tensor[:1])
        assert torch.equal(split_args[4][0], sample_tensor[:1])

    def test_chunkify_no_dummy_chunks(self):
        batch_size = 9
        num_chunks = 5

        sample_tensor = torch.arange(1, batch_size * 10 + 1).reshape(batch_size, 10)

        chunker = ldp.nn.TensorChunker(num_chunks=num_chunks)
        split_args, split_kwargs, dummy_chunk_flags = chunker.chunkify(sample_tensor)

        assert len(split_args) == num_chunks
        assert len(split_kwargs) == num_chunks
        assert dummy_chunk_flags == [False, False, False, False, False]
        assert torch.equal(split_args[0][0], sample_tensor[:2])
        assert torch.equal(split_args[1][0], sample_tensor[2:4])
        assert torch.equal(split_args[2][0], sample_tensor[4:6])
        assert torch.equal(split_args[3][0], sample_tensor[6:8])
        assert torch.equal(split_args[4][0], sample_tensor[8:])

    def test_chunkify_with_args_and_kwargs(self):
        batch_size = 2
        num_chunks = 3

        sample_tensor = torch.arange(1, batch_size * 10 + 1).reshape(batch_size, 10)
        sample_tensor_kwarg = torch.arange(1, batch_size * 5 + 1).reshape(batch_size, 5)
        sample_kwargs = {
            "key1": sample_tensor_kwarg,
            "key2": "Not split",
        }

        chunker = ldp.nn.TensorChunker(num_chunks=num_chunks)
        split_args, split_kwargs, dummy_chunk_flags = chunker.chunkify(
            sample_tensor, **sample_kwargs
        )

        assert len(split_args) == num_chunks
        assert len(split_kwargs) == num_chunks
        assert dummy_chunk_flags == [False, False, True]
        assert torch.equal(split_args[0][0], sample_tensor[:1])
        assert torch.equal(split_args[1][0], sample_tensor[1:2])
        assert torch.equal(split_args[2][0], sample_tensor[:1])
        assert torch.equal(split_kwargs[0]["key1"], sample_tensor_kwarg[:1])
        assert torch.equal(split_kwargs[1]["key1"], sample_tensor_kwarg[1:2])
        assert torch.equal(split_kwargs[2]["key1"], sample_tensor_kwarg[:1])
        assert all(split_kwargs[i]["key2"] == "Not split" for i in range(num_chunks))

    def test_dechunkify(self):
        batch_size = 2
        num_chunks = 3

        # Simulate outputs from handlers
        outputs_list = []
        for i in range(num_chunks):
            sequences = torch.tensor([[2 * i], [2 * i + 1]], dtype=torch.long)
            scores = None  # Simplify the test by not including scores
            output = GenerateDecoderOnlyOutput(sequences=sequences, scores=scores)  # type: ignore[arg-type]
            outputs_list.append(output)
        dummy_chunk_flags = [i >= batch_size for i in range(num_chunks)]

        # Dechunkify outputs
        combined_output = ldp.nn.TensorChunker.dechunkify(
            outputs_list, dummy_chunk_flags
        )

        # Since the last chunk was a dummy, it should be excluded
        expected_sequences = [
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([2]),
            torch.tensor([3]),
        ]  # Sequences from first two outputs
        assert all(
            starmap(
                torch.equal,
                zip(combined_output.sequences, expected_sequences, strict=True),
            )
        )
        assert combined_output.scores is None


def randomize_port(
    config: ldp.nn.ParallelModeConfig | None,
) -> ldp.nn.ParallelModeConfig | None:
    """If a test fails and is retried, reset the port to be safe."""
    if not config:
        return None
    return config.model_copy(update={"torch_port": get_unused_port()})


class TestHandlers:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("parallel_mode_config", PARALLEL_MODE_CONFIGS)
    @pytest.mark.usefixtures("seed_zero")
    async def test_generation(
        self,
        parallel_mode_config: ldp.nn.ParallelModeConfig | None,
    ) -> None:
        parallel_mode_config = randomize_port(parallel_mode_config)
        model_config = ldp.nn.LMConfig(
            model="gpt2",
            dtype=ldp.nn.TorchDType.bf16,
            # For this bug: https://github.com/huggingface/transformers/issues/31884
            # Only need to be careful in this case b/c we are checking the exact number
            # of tokens generated.
            tokenizer_args={"clean_up_tokenization_spaces": False},
        )
        # Need this for the tokenizer. TODO: see if we can remove it.
        model_config.resolve_model_location()
        tokenizer = model_config.get_tokenizer()

        handler = ldp.nn.TransformerHandlerConfig(
            lm_config=model_config,
            lm_type=ldp.nn.LMType.GENERATION,
            batch_size=4,
            module_call_fn=ldp.nn.AsyncTransformerInterface.model_generate,
            collate_fn=partial(
                ldp.nn.collate_fn_transformer_left_pad,
                pad_token_id=tokenizer.pad_token_id,
            ),
            decollate_fn=ldp.nn.decollate_fn_transformer_decoder,
            parallel_mode_config=parallel_mode_config,
        ).make_async_module()

        # Set large to encourage generation of end token and test uneven generations.
        max_new_tokens = 100
        outputs = await asyncio.gather(*[
            handler(
                in_text,
                temperature=1.0,
                max_new_tokens=max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=True,
            )
            for in_text in ("hello, ", "goodbye, ")
        ])

        for out_text, out_logits in outputs:
            # The length of out_logits must be:
            # * <= max_new_tokens (cannot exceed the max generation length).
            # * >= len(tokenizer(out_text)["input_ids"]).
            # The reason for >= and not == in the second condition is for rare
            # cases where tokenizer optimizes/shrinks tokens as we're re-tokenizing model outputs
            # in the below code.
            assert len(out_logits) <= max_new_tokens
            assert len(out_logits) >= len(tokenizer(out_text)["input_ids"])

    @pytest.mark.parametrize("parallel_mode_config", PARALLEL_MODE_CONFIGS)
    def test_state_dicts(
        self,
        parallel_mode_config: ldp.nn.ParallelModeConfig | None,
    ) -> None:
        parallel_mode_config = randomize_port(parallel_mode_config)
        model_config = ldp.nn.LMConfig(model="gpt2", dtype=ldp.nn.TorchDType.bf16)
        model_config.resolve_model_location()

        handler = ldp.nn.TransformerHandlerConfig(
            lm_config=model_config,
            lm_type=ldp.nn.LMType.GENERATION,
            batch_size=1,
            module_call_fn=ldp.nn.AsyncTransformerInterface.model_generate,
            collate_fn=partial(
                ldp.nn.collate_fn_transformer_left_pad,
                pad_token_id=model_config.get_tokenizer().pad_token_id,
            ),
            decollate_fn=ldp.nn.decollate_fn_transformer_decoder,
            parallel_mode_config=parallel_mode_config,
        ).make_async_module()

        orig_state_dict = handler.state_dict()
        zeros_state_dict = {k: torch.zeros_like(v) for k, v in orig_state_dict.items()}

        if not parallel_mode_config:
            # load_state_dict() is available
            handler.load_state_dict(zeros_state_dict)

        else:
            with pytest.raises(NotImplementedError):
                # check that we can't do this
                handler.load_state_dict(zeros_state_dict)

            # now check we can at least load from disk
            tempdir = Path(mkdtemp())
            tempfile = tempdir / "pytorch_model_fsdp.bin"
            torch.save(zeros_state_dict, tempfile)
            handler.load_checkpoint(tempdir)
            rmtree(tempdir)

        for k, v in handler.state_dict().items():
            assert torch.equal(v, zeros_state_dict[k])

    @pytest.mark.skipif(not TEST_GPUS, reason="Requires GPUs")
    @pytest.mark.parametrize("sharded", [True, False])
    def test_distributed_checkpoints(self, sharded: bool) -> None:
        model_config = ldp.nn.LMConfig(model="gpt2", dtype=ldp.nn.TorchDType.bf16)
        model_config.resolve_model_location()

        handler_config = ldp.nn.TransformerHandlerConfig(
            lm_config=model_config,
            lm_type=ldp.nn.LMType.GENERATION,
            batch_size=1,
            module_call_fn=ldp.nn.AsyncTransformerInterface.model_generate,
            collate_fn=partial(
                ldp.nn.collate_fn_transformer_left_pad,
                pad_token_id=model_config.get_tokenizer().pad_token_id,
            ),
            decollate_fn=ldp.nn.decollate_fn_transformer_decoder,
            parallel_mode_config=ldp.nn.ParallelModeConfig(
                num_workers=2,
                num_cpus_per_worker=1,
                state_dict_type=(
                    StateDictType.SHARDED_STATE_DICT
                    if sharded
                    else StateDictType.FULL_STATE_DICT
                ),
            ),
        )
        handler = ldp.nn.ParallelAsyncTransformer(handler_config)

        # CPU offloading before initializing the optimizer appears to be causing reshapes
        orig_state_dict = handler.state_dict(offload_to_cpu=False)

        # update the model
        @handler.wrap_func(worker_agg_fn=lambda x: sum(x) / len(x))
        def update_model(handler: ldp.nn.TransformerHandler) -> float:
            parameters = list(handler.module.parameters())
            opt = SGD(parameters, lr=0.1)
            x = torch.ones((2, 2), dtype=torch.long).to(handler.module.device)  # type: ignore[arg-type]
            loss = handler.module(x, labels=x).loss
            loss.backward()
            opt.step()
            return loss.item()

        update_model()
        updated_state_dict = handler.state_dict()
        # make sure at least one parameter has changed
        for k, v in updated_state_dict.items():
            if not torch.equal(v, orig_state_dict[k]):
                break
        else:
            raise ValueError("No parameter changed")

        ckpt_path = Path(mkdtemp())
        handler.save_checkpoint(ckpt_path)
        if sharded:
            # one per gpu
            assert (
                len(list((ckpt_path / "pytorch_model_fsdp_0").rglob("*.distcp"))) == 2
            )
        else:
            assert len(list(ckpt_path.rglob("*.bin"))) == 1  # should've been merged

        handler.teardown()

        # Make sure we can load the checkpoint
        new_handler = handler_config.model_copy(
            update={
                "checkpoint": ckpt_path,
                "parallel_mode_config.torch_port": get_unused_port(),
            }
        ).make_async_module()

        # finally, check we got updated_state_dict back
        assert all(
            torch.equal(v, updated_state_dict[k])
            for k, v in new_handler.state_dict().items()
        )

        # cleanup
        rmtree(ckpt_path)

    @pytest.mark.skipif(not TEST_GPUS, reason="Requires GPUs")
    def test_consistent_weights(self):
        # With cpu_ram_efficient_loading, we are asking HF to load the weights on one device
        # and then distribute them. Check that this gives us the same weights as the naive method.
        parallel_configs = {
            "single": None,
            "parallel": ldp.nn.ParallelModeConfig(
                num_workers=2, num_cpus_per_worker=1, cpu_ram_efficient_loading=True
            ),
        }

        model_config = ldp.nn.LMConfig(model="gpt2", dtype=ldp.nn.TorchDType.bf16)
        model_config.resolve_model_location()
        state_dicts = {}

        for mode, parallel_config in parallel_configs.items():
            handler_config = ldp.nn.TransformerHandlerConfig(
                lm_config=model_config,
                lm_type=ldp.nn.LMType.GENERATION,
                batch_size=1,
                module_call_fn=ldp.nn.AsyncTransformerInterface.model_generate,
                collate_fn=partial(
                    ldp.nn.collate_fn_transformer_left_pad,
                    pad_token_id=model_config.get_tokenizer().pad_token_id,
                ),
                decollate_fn=ldp.nn.decollate_fn_transformer_decoder,
                parallel_mode_config=parallel_config,
            )

            handler = handler_config.make_async_module()
            state_dicts[mode] = {k: v.cpu() for k, v in handler.state_dict().items()}

        for k, v in state_dicts["parallel"].items():
            assert torch.equal(v, state_dicts["single"][k])
