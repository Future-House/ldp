import contextvars
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import ModuleType
from uuid import UUID, uuid4

from aviary.core import is_coroutine_callable
from pydantic import BaseModel, field_serializer, field_validator


def _lazy_import_networkx() -> ModuleType:
    try:
        import networkx as nx
    except ImportError as e:
        raise ImportError(
            "networkx is required for compute graph operations. "
            "Please install it with: pip install ldp[scg]"
        ) from e
    return nx


def _lazy_import_tree() -> ModuleType:
    try:
        import tree
    except ImportError as e:
        raise ImportError(
            "tree is required for compute graph operations. "
            "Please install it with: pip install ldp[scg]"
        ) from e
    return tree


class CallID(BaseModel):
    run_id: UUID
    fwd_id: UUID

    def __repr__(self) -> str:
        return f"{self.run_id}:{self.fwd_id}"

    def __hash__(self) -> int:
        return hash((self.run_id, self.fwd_id))

    @field_validator("run_id", "fwd_id", mode="before")
    @classmethod
    def validate_uuid(cls, value: UUID | str) -> UUID:
        if isinstance(value, str):
            return UUID(value)
        return value

    @field_serializer("run_id", "fwd_id")
    def serialize_uuid(self, value: UUID) -> str:
        return str(value)

    def __init__(self, run_id: str | UUID, fwd_id: str | UUID):
        # Convenience so we can use positional arguments
        super().__init__(run_id=run_id, fwd_id=fwd_id)


_RUN_ID = contextvars.ContextVar[UUID]("run_id")
_CALL_ID = contextvars.ContextVar[CallID]("call_id")


@asynccontextmanager
async def compute_graph() -> AsyncIterator[UUID]:  # noqa: RUF029
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
async def op_call() -> AsyncIterator[CallID]:  # noqa: RUF029
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
    """Automatically set and reset the training_mode with a context manager."""

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

    def __enter__(self) -> None:
        self.prev_training_mode = get_training_mode()
        set_training_mode(self.training_mode)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        set_training_mode(self.prev_training_mode)

    async def __aenter__(self) -> None:
        self.__enter__()

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        self.__exit__(exc_type, exc_value, traceback)


train_mode = _TrainingModeContext(training_mode=True)
eval_mode = _TrainingModeContext(training_mode=False)
