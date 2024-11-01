import contextlib
import json
import sys
from typing import Any, Coroutine

from aviary.core import Message, Tool, ToolCall, ToolRequestMessage

from ldp.graph import IdentityOp, OpResult, compute_graph

from .agent import Agent, TAgentState
from .simple_agent import SimpleAgentState

MISSING_DEFAULT = object()
# Clears the current input and allows the user to start over, e.g. if they made a mistake
CLEAR = "CLEAR"
# Exits the agent by raising a RuntimeError. Makes it possible to interrupt a rollout
EXIT = "EXIT"


class InteractiveAgent(Agent[SimpleAgentState]):
    """An "agent" that provides an interface for human users to interact with environments."""

    def __init__(self):
        self.action_op = IdentityOp[ToolRequestMessage]()

    async def init_state(self, tools: list[Tool]) -> SimpleAgentState:
        print()  # add a newline to flush any progress bars, etc
        print("AVAILABLE TOOLS:\n" + ("-" * 80))
        for tool in tools:
            info = tool.info
            docstring = f"{info.name}("
            arg_docs = []
            for pname, pprops in info.parameters.properties.items():
                docstring += self._get_param_string(pname, pprops) + ", "
                arg_doc = "   " + pname
                if "description" in pprops:
                    arg_doc += ": " + pprops.get("description", "")
                arg_docs.append(arg_doc)

            docstring = docstring.rstrip(", ") + "):\n"
            docstring += "   " + info.description + "\n\n"
            docstring += "\n".join(arg_docs)
            docstring += "\n"
            print(docstring)
        print("-" * 80)
        return SimpleAgentState(tools=tools)

    @compute_graph()
    async def get_asv(  # noqa: C901
        self, agent_state: SimpleAgentState, obs: list[Message]
    ) -> tuple[OpResult[ToolRequestMessage], SimpleAgentState, float]:
        print()  # add a newline to flush any progress bars, etc
        print("OBSERVATIONS:\n" + ("-" * 80))
        for msg in obs:
            print((msg.content or "") + "\n")
        print("-" * 80)

        next_agent_state = agent_state.get_next_state(obs)

        tool: Tool | None = None
        while not tool:
            tool_choice = input(">>> Select tool by name: ")
            if tool_choice == CLEAR:
                continue
            if tool_choice == EXIT:
                raise RuntimeError("User requested to kill the agent.")

            tool = next(
                (t for t in agent_state.tools if t.info.name == tool_choice), None
            )
            if not tool:
                print(
                    f"Tool '{tool_choice}' not found. Please select from the above"
                    " tools."
                )

        params: dict[str, Any] = {}
        for pname, pprops in tool.info.parameters.properties.items():
            pdefault = pprops.get("default", MISSING_DEFAULT)
            prompt = f">>> Enter parameter ({self._get_param_string(pname, pprops)}): "
            while True:
                value = input(prompt)
                if value == CLEAR:
                    return await self.get_asv(agent_state, obs)  # just start over
                if value == EXIT:
                    raise RuntimeError("User requested to kill the agent.")

                with contextlib.suppress(json.JSONDecodeError):
                    # lets us load ints, etc. Otherwise, assume it's a string
                    value = json.loads(value)
                if not value:
                    if pdefault is MISSING_DEFAULT:
                        print("Parameter is required.")
                        continue

                    value = pdefault

                params[pname] = value
                break

        tool_call = ToolCall.from_tool(tool, **params)
        action = await self.action_op(ToolRequestMessage(tool_calls=[tool_call]))

        next_agent_state.messages = [*next_agent_state.messages, action.value]

        return action, next_agent_state, 0.0

    @staticmethod
    def _get_param_string(pname: str, pprops: dict[str, Any]) -> str:
        pstring = pname
        if ptype := (pprops.get("type") or "Any"):
            pstring += f": {ptype}"

        if (pdefault := pprops.get("default", MISSING_DEFAULT)) is not MISSING_DEFAULT:
            pstring += f" = {pdefault}"
        return pstring


@contextlib.contextmanager
def trace_function(target_func):
    """A context manager that injects a breakpoint in the target function."""

    def trace_calls(frame, event, arg):
        if event == "call" and frame.f_code == target_func.__code__:
            sys.breakpointhook(frame)
        return trace_calls

    sys.settrace(trace_calls)
    try:
        yield
    finally:
        sys.settrace(None)


class DebugAgent(Agent[TAgentState]):
    """An "agent" that wraps another agent and injects breakpoints in init_state and get_asv."""

    def __init__(self, agent: Agent[TAgentState]):
        self.agent = agent

    async def init_state(self, tools: list[Tool]) -> TAgentState:
        with trace_function(self.agent.init_state):
            return await self.agent.init_state(tools)

    async def get_asv(
        self, agent_state: TAgentState, obs: list[Message]
    ) -> tuple[OpResult[ToolRequestMessage], TAgentState, float]:
        with trace_function(self.agent.get_asv):
            return await self.agent.get_asv(agent_state, obs)
