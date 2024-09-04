import contextvars
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import NamedTuple
from uuid import UUID, uuid4

from aviary.utils import is_coroutine_callable


class CallID(NamedTuple):
    run_id: UUID
    fwd_id: UUID

    def __repr__(self) -> str:
        return f"{self.run_id}:{self.fwd_id}"


_RUN_ID = contextvars.ContextVar[UUID]("run_id")
_CALL_ID = contextvars.ContextVar[CallID]("call_id")


@asynccontextmanager
async def compute_graph() -> AsyncIterator[UUID]:
    """Initialize a compute graph by setting a run ID.

    If a run ID is already set (i.e. we are already inside a
    get_run_id() context), then the existing run ID is returned.
    Otherwise, a new UUID is created.
    """
    try:
        # If a run ID is set, return it.
        run_id = _RUN_ID.get()
        token: contextvars.Token | None = None
    except LookupError:
        # If not, make a new run ID.
        run_id = uuid4()
        token = _RUN_ID.set(run_id)

    try:
        yield run_id
    finally:
        if token is not None:
            # token is not None if we made a new run ID. In that case,
            # reset the context to its previous state.
            _RUN_ID.reset(token)


def get_run_id() -> UUID:
    """Get the current run ID."""
    try:
        return _RUN_ID.get()
    except LookupError:
        raise RuntimeError(
            "Attempting to access run ID, but not inside compute graph context."
        ) from None


@asynccontextmanager
async def op_call() -> AsyncIterator[CallID]:
    """Decorate an op call with a call ID.

    If a call ID is already set (i.e. we are already inside an op call),
    then the existing call ID is returned.
    Otherwise, a new UUID is created.
    """
    # Get run_id in case we need to construct a CallID, but this also serves
    # as a check that we're inside compute_graph()
    run_id = get_run_id()

    try:
        call_id = _CALL_ID.get()
        token: contextvars.Token | None = None
    except LookupError:
        fwd_id = uuid4()
        call_id = CallID(run_id, fwd_id)
        token = _CALL_ID.set(call_id)

    try:
        yield call_id
    finally:
        if token is not None:
            # token is not None if we made a new call ID. In that case,
            # reset the context to its previous state.
            _CALL_ID.reset(token)


def get_call_id() -> CallID:
    """Get the current call ID."""
    try:
        return _CALL_ID.get()
    except LookupError:
        raise RuntimeError(
            "Attempting to access call ID, but not inside op call context."
        ) from None


_TRAINING_MODE = contextvars.ContextVar[bool]("training_mode", default=True)


def get_training_mode() -> bool:
    """Get the current training mode."""
    return _TRAINING_MODE.get()


def set_training_mode(training_mode: bool) -> None:
    """Set the training mode."""
    _TRAINING_MODE.set(training_mode)


class _TrainingModeContext:
    def __init__(self, training_mode: bool):
        self.training_mode = training_mode
        self.prev_training_mode = get_training_mode()

    def __call__(self, fn=None):
        if fn is None:
            return self

        if is_coroutine_callable(fn):

            async def wrapper(*args, **kwargs):
                async with self:
                    return await fn(*args, **kwargs)

        else:

            def wrapper(*args, **kwargs):
                with self:
                    return fn(*args, **kwargs)

        return wrapper

    def __enter__(self):
        self.prev_training_mode = get_training_mode()
        set_training_mode(self.training_mode)

    def __exit__(self, exc_type, exc_value, traceback):
        set_training_mode(self.prev_training_mode)

    async def __aenter__(self):
        self.__enter__()

    async def __aexit__(self, exc_type, exc_value, traceback):
        self.__exit__(exc_type, exc_value, traceback)


train_mode = _TrainingModeContext(training_mode=True)
eval_mode = _TrainingModeContext(training_mode=False)