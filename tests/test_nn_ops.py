import asyncio

import pytest
from aviary.core import Message

import ldp.nn
from ldp.graph import OpResult


class TestLocalLLMCallOp:
    @pytest.mark.asyncio
    async def test_batching_consistent_results(self):
        """Tests that padding in batched calls does not affect the results (attention_mask is set correctly)."""
        model_config = ldp.nn.LMConfig(
            model="gpt2", device="cpu", dtype=ldp.nn.TorchDType.fp32
        )
        local_llm_call_op = ldp.nn.LocalLLMCallOp(
            model_config, batch_size=2, max_wait_interval=1.0
        )

        messages = [
            Message(content=text)
            for text in ("Hello, how are you?", "Hello, how are you?")
        ]

        async def forward_batch() -> list[OpResult[Message]]:
            return await asyncio.gather(*[
                local_llm_call_op([msg], temperature=1.0, max_new_tokens=10)
                for msg in messages
            ])

        # First forward batch with seed set to 0
        ldp.nn.set_seed(0)
        results_first_call = await forward_batch()

        # Re-seed and forward batch again with padding to check for consistency
        ldp.nn.set_seed(0)
        messages[
            -1
        ].content = (
            "Some very long text that would create lots of padding in the batch."
        )
        results_second_call = await forward_batch()

        assert len(results_first_call) == len(results_second_call), (
            "Expected the number of results to match between the two calls."
        )
        assert results_first_call[0].value == results_second_call[0].value, (
            "Expected the results to match between the two calls, but got differing"
            " results."
        )
