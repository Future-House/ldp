import asyncio
import random
from uuid import UUID

import pytest

from ldp.graph import (
    CallID,
    compute_graph,
    eval_mode,
    get_call_id,
    get_run_id,
    get_training_mode,
    op_call,
    set_training_mode,
    train_mode,
)
from ldp.graph.op_utils import _RUN_ID  # noqa: PLC2701


@pytest.mark.asyncio
async def test_run_ids():
    # check that we don't create a new run ID if we're
    # already in a context
    async with compute_graph():
        run_id_1 = get_run_id()
        async with compute_graph():
            run_id_2 = get_run_id()
            assert run_id_1 == run_id_2, (
                "Should not create a new run ID if already in a run context."
            )

    # Check that after exiting the context, _RUN_ID is no longer set
    with pytest.raises(LookupError):
        _RUN_ID.get()

    # Now check that we don't clobber in coroutines
    async def run_id_test() -> UUID:
        async with compute_graph():
            # Wait randomly, giving time for other calls to potentially clobber
            # this one.
            await asyncio.sleep(random.uniform(0.1, 0.5))
            return get_run_id()

    run_ids = await asyncio.gather(*[run_id_test() for _ in range(10)])
    assert len(run_ids) == len(set(run_ids)), (
        "At least two compute graphs had the same run ID, which indicates a clobber."
    )


@pytest.mark.asyncio
async def test_call_ids():
    # check that we don't create a new call ID if we're
    # already in a context
    async with compute_graph(), op_call():
        call_id_1 = get_call_id()
        async with op_call():
            call_id_2 = get_call_id()
            assert call_id_1 == call_id_2, (
                "Should not create a new call ID if already in a call context."
            )

    # Check we cannot create a call ID if not in a run context
    with pytest.raises(RuntimeError, match=r".*not inside compute graph context.*"):
        async with op_call():
            pass

    async def call_test() -> CallID:
        async with op_call():
            # Wait randomly, giving time for other calls to potentially clobber
            # this one.
            await asyncio.sleep(random.uniform(0.1, 0.5))
            return get_call_id()

    async with compute_graph():
        call_ids = await asyncio.gather(*[call_test() for _ in range(10)])
    assert len(call_ids) == len(set(call_ids)), (
        "At least two compute graphs had the same run ID, which indicates a clobber."
    )


@pytest.mark.asyncio
async def test_training_mode():
    with eval_mode():
        assert not get_training_mode(), "Training mode was not set to False"
    assert get_training_mode(), (
        "Training mode should have been reset to True after exiting context"
    )

    set_training_mode(False)
    with train_mode():
        assert get_training_mode(), "Training mode was not set to True"
    assert not get_training_mode(), (
        "Training mode should have been reset to False after exiting context"
    )

    # Put back to training for next round of tests
    set_training_mode(True)

    async def training_mode_test(i: int):
        train = i % 2 == 0
        set_training_mode(train)
        # wait a random amount of time to give a chance for other coroutines to clobber
        await asyncio.sleep(random.uniform(0.1, 0.5))

        assert get_training_mode() == train, "Training mode was overwritten."

    await asyncio.gather(*[training_mode_test(i) for i in range(10)])
    # Make sure training_mode_test didn't change our training mode
    assert get_training_mode()

    # Make sure nesting works
    with train_mode():
        assert get_training_mode()
        with eval_mode():
            assert not get_training_mode()
            with train_mode():
                assert get_training_mode()
            assert not get_training_mode()
        assert get_training_mode()

    # Make sure modes are set correctly in coroutines

    @train_mode()
    async def train_then_eval():
        assert get_training_mode()
        await asyncio.sleep(0.0)
        async with eval_mode():
            assert not get_training_mode()

    @eval_mode()
    async def eval_then_train():
        assert not get_training_mode()
        await asyncio.sleep(0.0)
        async with train_mode():
            assert get_training_mode()

    await asyncio.gather(train_then_eval(), eval_then_train())
