import pytest

from ldp.alg.optimizer.replay_buffers import CircularReplayBuffer


def test_circular_buffer():
    buf = CircularReplayBuffer()

    samples = [{"state": 1, "action": 2, "reward": 3, "t": t} for t in range(5)]
    buf += samples
    buf.resize(3)  # should eject t=0, 1
    assert {sample["t"] for sample in buf} == {2, 3, 4}

    # check we can iterate
    next(buf.batched_iter(batch_size=3))

    # add a bad sample
    buf.append({})
    with pytest.raises(
        RuntimeError, match="Found buffer element with inconsistent keys"
    ):
        next(buf.batched_iter(batch_size=4))
