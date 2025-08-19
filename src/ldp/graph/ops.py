"""This module defines the Op class and its helper classes."""

from __future__ import annotations

import inspect
import itertools
import logging
import secrets
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeAlias, TypeVar
from uuid import UUID

if TYPE_CHECKING:
    import networkx as nx
    import tree
from pydantic import BaseModel, Field

from .op_utils import (
    CallID,
    _lazy_import_networkx,
    _lazy_import_tree,
    compute_graph,
    get_call_id,
    get_training_mode,
    op_call,
)

logger = logging.getLogger(__name__)


GradOutType: TypeAlias = (
    "tree.Structure | None"  # None means the gradient has terminated
)
GradInType: TypeAlias = tuple[Sequence[GradOutType], Mapping[str, GradOutType]]
BackwardsType: TypeAlias = Callable[
    # Call signature of Op.backward
    ["OpCtx", list, dict, "tree.Structure", CallID],
    GradInType,
]
TOutput_co = TypeVar("TOutput_co", covariant=True)


class OpResult(Generic[TOutput_co]):
    """Result of a forward pass, used in the compute graph."""

    def __init__(
        self, call_id: CallID | Any, op_name: str, op_class_name: str, value: TOutput_co
    ):
        """
        Initialize an OpResult instance.

        Args:
            call_id: The unique identifier for the call.
            op_name: Name of the Op instance (i.e. op.name) that produced this OpResult.
            op_class_name: Fully qualified name of the class of the Op that produced this OpResult.
            value: The output of the call.
        """
        self.call_id = CallID.model_validate(call_id)
        self.op_name = op_name
        self.op_class_name = op_class_name
        self.value = value

    def to_dict(self) -> dict[str, Any]:
        value_dump = (
            self.value.model_dump() if isinstance(self.value, BaseModel) else self.value
        )

        return {
            "call_id": self.call_id.model_dump(),
            "op_name": self.op_name,
            "op_class_name": self.op_class_name,
            "value": value_dump,
            "logprob": self.logprob,
        }

    @classmethod
    def from_dict(
        cls, t_output: type[TOutput_co], dump: dict[str, Any]
    ) -> OpResult[TOutput_co]:
        value = dump.pop("value")
        if issubclass(t_output, BaseModel):
            value = t_output.model_validate(value)

        # logprob was added to serialization for convenience,
        # but is not actually an OpResult argument, so we just discard it here
        dump.pop("logprob", None)
        return cls[t_output](**dump, value=value)  # type: ignore[index]

    def __hash__(self) -> int:
        return hash(self.call_id)

    def compute_grads(
        self,
        grad_output: tree.Structure | None = None,
        backward_fns: Mapping[str | type, BackwardsType] | None = None,
    ) -> None:
        """
        Compute the gradient of the backward graph in-place.

        This executes topological traversal.
        It is up to the Op to:
            (a) define the backward computation
            (b) store internal gradients for optimizer updates.
        """
        tree = _lazy_import_tree()

        # call ID -> [d op(x) / d x] for each op that consumes x[
        # due to interaction between ruff and mypy, we need type ignore
        grad_outputs: dict[CallID, list[tree.Structure]] = defaultdict(list)  # type: ignore[name-defined]

        # grad_outputs stores a list of output grads (corresponding to each consuming op call).
        # Since the root node is not consumed by any other node, we create a singleton list here.
        # If None was passed, set it to 0 so that we don't prune the compute graph here.
        grad_outputs[self.call_id] = [grad_output] if grad_output is not None else [0.0]

        # We will traverse the graph in reverse topological order
        for node in self.traverse():
            # get output gradients
            grad_output = grad_outputs[node.call_id]
            if not grad_output:
                # compute graph terminated
                continue
            # Make sure structure of grads match before summing
            try:
                [tree.assert_same_structure(grad_output[0], g) for g in grad_output[1:]]
            except ValueError as e:
                raise ValueError(
                    "Mismatched gradient structures in compute graph for at Op:"
                    f" {self.op_name}."
                ) from e
            aggregated_grad_output = tree.map_structure(lambda *x: sum(x), *grad_output)  # noqa: FURB111

            input_args, input_kwargs = node.inputs
            arg_grads, kwarg_grads = node._run_backward(
                input_args,
                input_kwargs,
                aggregated_grad_output,
                node._resolve_backward_impl(backward_fns),
            )

            for a, g in zip(input_args, arg_grads, strict=True):
                # Must have exact match between input_args and arg_grads
                # Only propagate gradients to input OpResults if grad is not None
                if g is not None and isinstance(a, OpResult):
                    grad_outputs[a.call_id].append(g)

            if kwarg_grads.keys() != input_kwargs.keys():
                raise ValueError(
                    "Mismatch between grads returned in Op.backward and its input"
                    f" kwargs. Expected {input_kwargs.keys()}, got"
                    f" {kwarg_grads.keys()}."
                )
            for k, a in input_kwargs.items():
                # input_kwargs.keys() may be a subset of kwarg_grads.keys() if defaults
                # are specified
                if (g := kwarg_grads[k]) is not None and isinstance(a, OpResult):
                    grad_outputs[a.call_id].append(g)

    def _resolve_backward_impl(
        self, backward_fns: Mapping[str | type, BackwardsType] | None
    ) -> BackwardsType:
        backward_fns = backward_fns or {}
        for key in (self.ctx.op_name, self.op_class_name, self.op_class):
            if key in backward_fns:
                return backward_fns[key]
        return self.op_class.backward

    def _run_backward(
        self,
        input_args: list[ResultOrValue],
        input_kwargs: dict[str, ResultOrValue],
        grad_output: tree.Structure,
        backward_fn: BackwardsType,
    ) -> GradInType:
        self._update_ctx("grad_output", grad_output)
        unwrapped_input_args = [OpResult.unwrap_value(a) for a in input_args]
        unwrapped_input_kwargs = {
            k: OpResult.unwrap_value(v) for k, v in input_kwargs.items()
        }
        input_grads = backward_fn(
            self.ctx,
            unwrapped_input_args,
            unwrapped_input_kwargs,
            grad_output,
            self.call_id,
        )
        self._update_ctx("grad_input", input_grads)
        return input_grads

    @property
    def op_class(self) -> type[Op]:
        return _OP_CLASS_REGISTRY[self.op_class_name]

    @property
    def ctx(self) -> OpCtx:
        # This is a property to avoid serialization of the context. There are two reasons:
        # 1. Contexts have their own persist() mechanism for serialization
        # 2. We'd prefer contexts to be created via get_or_create(). Allowing for arbitrary
        #    deserialization makes it hard to enforce that.
        return OpCtx.get_or_create(self.op_name)

    @property
    def op(self) -> Op:
        return _OP_REGISTRY[self.op_name]

    def get_compute_graph(self, backward: bool = True) -> nx.DiGraph:
        """Construct a directed graph of the compute graph that led to this OpResult.

        Args:
            backward: If True (default), constructs the backwards graph in which outputs
                point to inputs. If False, constructs the forward call graph.
                For most cases (e.g. backprop), backward=True is desirable.

        Returns:
            A digraph in which nodes are OpResults.
        """

        def add_edges(graph: nx.DiGraph, node: OpResult) -> None:
            """Recursively add edges to the input graph."""
            input_args, input_kwargs = node.inputs
            for x in itertools.chain(input_args, input_kwargs.values()):
                if isinstance(x, OpResult):
                    edge = (node, x) if backward else (x, node)
                    graph.add_edge(*edge)
                    add_edges(graph, x)

        graph = _lazy_import_networkx().DiGraph()
        graph.add_node(self)
        add_edges(graph, self)

        return graph

    def get_upstream_results(self, op: str | Op) -> Iterator[OpResult]:
        """Get all OpResults upstream of this node that were produced by the given Op.

        Args:
            op: Will return all upstream nodes that were produced by this Op. Can provide
                 an instance or Op name.
        """
        if isinstance(op, Op):
            op = op.name

        return self.traverse(filter_fn=lambda node: node.op_name == op)

    def traverse(
        self,
        topological_order: bool = True,
        filter_fn: Callable[[OpResult], bool] = lambda _: True,
    ) -> Iterator[OpResult]:
        """Traverse the compute graph that led to this OpResult.

        Args:
            topological_order: If True, traverse the backwards graph in topological
                order. This requires having the whole graph in memory. If False,
                traverse the backwards graph in depth-first order. This can be done
                lazily and is useful if we are trying to hydrate the graph node-by-node.
                Most user-facing cases can leave this as True. Defaults to True.
            filter_fn: Will only yield nodes that pass this filter function. Note that
                nodes that fail will still be traversed.

        Yields:
            An iterator over the nodes of this graph.
        """
        if topological_order:
            G = self.get_compute_graph()
            for node in _lazy_import_networkx().topological_sort(G):
                if filter_fn(node):
                    yield node

        else:
            # If not topological order, do a recursive depth-first traversal.
            # Note that, when traversing a node, its children do not need to be available
            # yet. This allows us to lazily load nodes when hydrating from a ctx backend.
            if filter_fn(self):
                yield self
            input_args, input_kwargs = self.inputs
            for a in itertools.chain(input_args, input_kwargs.values()):
                if isinstance(a, OpResult):
                    # Recursively apply depth-first traversal on each node
                    yield from a.traverse(topological_order=False)

    @property
    def inputs(self) -> tuple[list[ResultOrValue], dict[str, ResultOrValue]]:
        return self._get_from_ctx("input")

    @property
    def logprob(self) -> float | None:
        return self._get_from_ctx("logprob", default=None)

    @property
    def grad(self) -> tree.Structure | None:
        """Returns `d ln(P_{compute_graph}) / d self` or None if gradients have not been computed."""
        return self._get_from_ctx("grad_output")

    @property
    def run_id(self) -> UUID:
        return self.call_id.run_id

    @staticmethod
    def unwrap_value(result: ResultOrValue[TOutput_co]) -> TOutput_co:
        if isinstance(result, OpResult):
            return result.value
        return result

    def __repr__(self) -> str:
        return (
            f"OpResult(op={self.op_class_name}:{self.op_name}, "
            f"call_id={self.call_id}, value={self.value!r})"
        )

    def __str__(self) -> str:
        return str(self.value)

    def _get_from_ctx(self, key: str, **kwargs):
        if self.call_id is None:
            raise ValueError(
                "Attempting to access context but compute graph "
                "is not available for this OpResult."
            )

        return self.ctx.get(call_id=self.call_id, key=key, **kwargs)

    def _update_ctx(self, key: str, value: Any):
        if self.call_id is None:
            raise RuntimeError(
                "Attempting to update context but compute graph "
                "is not available for this OpResult."
            )

        self.ctx.update(call_id=self.call_id, key=key, value=value)


ResultOrValue: TypeAlias = OpResult[TOutput_co] | TOutput_co

# Sentinel value for get() default
NOT_FOUND = object()


class OpCtx(BaseModel):
    # A global registry of contexts. We'd prefer to use an existing context
    # for an Op if it already has been created. Also useful for persist_all()
    # NOTE: see the comment on _OP_REGISTRY below on a potential memory leak.
    _CTX_REGISTRY: ClassVar[dict[str, OpCtx]] = {}

    op_name: str

    data: dict = Field(
        default_factory=lambda: defaultdict(dict),
        exclude=True,
        description=(
            "Maps run_id -> (fwd_id, key) -> value. "
            "data is excluded from model_dump() etc because we do "
            "not use Pydantic to persist context information. That "
            "should be done via the DB backend instead. OpCtx will "
            "serialize op_name, which is enough to rehydrate "
            "from the DB."
        ),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._CTX_REGISTRY[self.op_name] = self

    @classmethod
    def get_or_create(cls, op_name: str) -> OpCtx:
        """Return an OpCtx corresponding to the Op with the given name."""
        try:
            return cls._CTX_REGISTRY[op_name]  # Get
        except KeyError:
            return cls(op_name=op_name)  # Create

    @classmethod
    def clear_contexts(cls, op_names: Iterable[str] | None = None) -> None:
        """Clear the data in all contexts. If op_names is provided, only clear those contexts."""
        if op_names is None:
            op_names = cls._CTX_REGISTRY.keys()
        for op_name in op_names:
            try:
                cls._CTX_REGISTRY[op_name].data.clear()
            except KeyError:
                logger.warning(f"Op with name={op_name} not found in context registry.")

    def get(self, call_id: CallID, key: str, default: Any = NOT_FOUND) -> Any:
        """Get an attribute with an optional default, emulating dict.get."""
        value = self.data.get(call_id.run_id, {}).get((call_id.fwd_id, key), default)
        if value is NOT_FOUND:
            raise KeyError(f"call_id={call_id}, key='{key}' not found in context")
        return value

    def update(self, call_id: CallID, key: str, value: Any):
        self.data[call_id.run_id][call_id.fwd_id, key] = value

    def get_input_grads(self, call_id: CallID) -> GradInType:
        # TODO: this function name is confusing. Let's deprecate it. We only use it
        # in tests as far as I can tell.
        try:
            return self.get(call_id, "grad_input")
        except KeyError as exc:
            raise ValueError(
                f"No gradients have been computed for call_id={call_id}."
            ) from exc


def resolve_fully_qualified_name(cls: type) -> str:
    return f"{cls.__module__}.{cls.__name__}"


# A global registry of Op classes, so we can look up backward() implementations
# without needing an instantiated Op.
_OP_CLASS_REGISTRY: dict[str, type[Op]] = {}

# A global registry of Op instances, so that OpResults can trace back their provenance
# TODO: this can leak memory if we are frequently deleting Ops. We should add a custom
# garbage collection hook to remove Ops from the registry once the reference count
# drops to 1 (i.e. just the registry).
_OP_REGISTRY: dict[str, Op] = {}


class Op(ABC, Generic[TOutput_co]):
    """
    An operation that is 'differentiable' and can be used in an optimizer.

    Think torch.autograd.Function that can also be applied to non-differentiable
    operations like prompt template formatting or Python function calls.

    These form a forward computation graph when composed with other Ops via
    __call__. In training mode, this graph is constructed dynamically.
    """

    # Name is not guaranteed to be unique. Reasons:
    # 1. We definitely don't want it to be unique when recreating a compute graph
    #    for training on previously-collected data. In that case, we want the Op's
    #    name to be the same as it was before, to match up contexts/OpResults
    # 2. Uniqueness could make some DB lookups faster, but I don't think we run the
    #    risk of OpCtx clobbers as long as call_id (which is guaranteed to be unique)
    #    is always used as part of the key.
    name: str
    ctx: OpCtx
    _fwd_args: list[inspect.Parameter]

    def clear_ctx(self) -> None:
        self.ctx.data.clear()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _OP_CLASS_REGISTRY[resolve_fully_qualified_name(cls)] = cls

    def __new__(cls, *args, **kwargs):  # noqa: ARG004
        instance = super().__new__(cls)

        # Needs to be overridden by caller if this Op is to have
        # an identifiable name in the compute graph. c.f. Agent.__init_subclass__
        # for an example of how to do this.
        instance.set_name(cls._make_unique_default_name())

        # Set an attribute to help us map positional forward arguments to parameter
        # names, for the backward pass. We do this on the instance and not cls b/c
        # some instancees may override (e.g FxnOp).
        fwd_sig = inspect.signature(instance.forward)
        instance._fwd_args = list(fwd_sig.parameters.values())

        return instance

    @classmethod
    def _make_unique_default_name(cls) -> str:
        # 6 bytes results in string of size 12
        return f"{cls.__name__}-{secrets.token_hex(6)}"

    def set_name(self, name: str) -> None:
        if _OP_REGISTRY.get(getattr(self, "name", "")) is self:
            # de-register before setting the new name
            del _OP_REGISTRY[self.name]

        self.name = name
        self.ctx = OpCtx.get_or_create(name)
        _OP_REGISTRY[name] = self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} (name={self.name}, id={id(self)})"

    @abstractmethod
    async def forward(self, *args, **kwargs) -> TOutput_co:
        """
        Forward pass of the Op. Must accept call_id as an argument.

        Returns:
            Depending on this Op's purpose, the return may be considered an action
                (e.g. a tool call) or it may not (e.g. a loss calculation).
        """

    @classmethod
    @abstractmethod
    def backward(
        cls,
        ctx: OpCtx,
        input_args: list[ResultOrValue],
        input_kwargs,
        grad_output: tree.Structure,
        call_id: CallID,
    ) -> GradInType:
        """
        Backward pass of the Op.

        Args:
            ctx: Context that was used during the forward pass.
            input_args: Variable-length input arguments passed to forward, i.e.
                via *args.
            input_kwargs: All other arguments passed to forward pass.
            grad_output: A list of backpropagated gradients from each consumer
                of the output of the forward pass. It is up to the implementation
                to decide how to aggregate these gradients (e.g. in most cases summing).
            call_id: Call ID of the forward pass.

        Returns:
            grad_input: `d log(p) / d input` for each input to the forward pass.
                It should include gradients for all input positional and keyword
                arguments. Set to None for gradients that should terminate.
        """

    def get_call_ids(self, run_ids: Collection[UUID] | None = None) -> set[CallID]:
        ctx = self.ctx
        if run_ids is None:
            run_ids = ctx.data.keys()

        # de-duplicate before constructing CallIDs
        ids = {(run_id, fwd_id) for run_id in run_ids for fwd_id, _ in ctx.data[run_id]}
        return set(itertools.starmap(CallID, ids))

    # This compute_graph() decoration will do nothing if we are already inside a compute graph.
    # We add it here in case we are calling a bare op(), in which case we want a graph
    # with a single node.
    @compute_graph()
    @op_call()
    async def __call__(self, *args, **kwargs) -> OpResult[TOutput_co]:
        call_id = get_call_id()

        if not all(
            arg.call_id.run_id == call_id.run_id
            for arg in itertools.chain(args, kwargs.values())
            if isinstance(arg, OpResult)
        ):
            raise RuntimeError(
                "All args and kwargs must have the same run_id as the call_id's run_id."
                " Consider using @compute_graph() decorator to ensure this."
            )

        # we're over-saving here - can explore later if memory usage is high
        # unpack the args and kwargs from the result holders
        unpacked_args = [(a.value if isinstance(a, OpResult) else a) for a in args]
        unpacked_kwargs = {
            k: v.value if isinstance(v, OpResult) else v for k, v in kwargs.items()
        }

        if get_training_mode():
            # If training, save the inputs for the backward pass
            # Map positional arguments to keyword arguments to make backward pass easier
            for i_arg, (arg, param) in enumerate(
                # strict=False b/c not all params in _fwd_args will be in args (i.e. defaults and **kwargs)
                zip(args, self._fwd_args, strict=False)
            ):
                # Don't need to check for too many args or collisions with kwargs, since forward()
                # will raise an exception anyway
                if param.kind == inspect.Parameter.VAR_POSITIONAL:
                    # *args, so scoop up the rest of the arg tuple.
                    var_args = list(args[i_arg:])
                    break

                # Normal positional arg
                kwargs[param.name] = arg
            else:
                var_args = []  # there were no *args if we got here

            self.ctx.update(call_id, "input", (var_args, kwargs))

        # actually call forward pass with unpacked args and kwargs
        result = await self.forward(*unpacked_args, **unpacked_kwargs)
        t_output: type[TOutput_co] = type(result)

        # Now package up my result so it can be consumed by other calls.
        # Explicitly specify t_output. OpResult[TOutput] returns a generic object
        op_result = OpResult[t_output](  # type: ignore[valid-type]
            value=result,
            call_id=call_id,
            op_name=self.name,
            op_class_name=resolve_fully_qualified_name(type(self)),
        )

        if get_training_mode():
            self.ctx.update(call_id, "output", op_result)

        return op_result

    def get_input_grads(self, call_id: CallID) -> GradInType:
        return self.ctx.get_input_grads(call_id)
