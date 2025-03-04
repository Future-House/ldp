import pytest

from ldp.alg.optimizer.replay_buffers import (
    CircularReplayBuffer,
    PrioritizedReplayBuffer,
)


@pytest.mark.asyncio
async def test_circular_buffer() -> None:
    buf = CircularReplayBuffer()

    samples = [{"state": 1, "action": 2, "reward": 3, "t": t} for t in range(5)]
    buf += samples
    buf.resize(3)  # should eject t=0, 1
    assert {sample["t"] for sample in buf} == {2, 3, 4}

    await buf.prepare_for_sampling()

    # check we can iterate
    next(buf.batched_iter(batch_size=3))

    # add a bad sample
    buf.append({})
    with pytest.raises(
        RuntimeError, match="Found buffer element with inconsistent keys"
    ):
        next(buf.batched_iter(batch_size=4))

    buf.clear()
    assert not buf, "Failed to clear data"


async def _dummy_q_function(*args, **kwargs) -> float:  # noqa: ARG001, RUF029
    return 1.0


@pytest.mark.asyncio
async def test_prioritized_buffer():
    buf = PrioritizedReplayBuffer(alpha=1, ranked=False, q_function=_dummy_q_function)

    buf += [
        {
            "input_args": (),
            "input_kwargs": {},
            "discounted_return": -1.0 if t % 2 else 1.0,
            "t": t,
        }
        for t in range(6)
    ]
    buf.resize(3)

    with pytest.raises(RuntimeError, match="TD errors not available"):
        next(buf.batched_iter(batch_size=3))

    await buf.prepare_for_sampling()

    # check we can iterate
    batch = next(buf.batched_iter(batch_size=3))

    # The odd timesteps should have priority because they have higher error
    assert all(t % 2 for t in batch["t"])
