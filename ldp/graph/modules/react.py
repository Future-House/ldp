import ast
import re
import textwrap
from collections.abc import Iterable
from enum import StrEnum
from typing import Any

from aviary.message import EMPTY_CONTENT_BASE_MSG, MalformedMessageError, Message
from aviary.tools import Tool, ToolCall, ToolRequestMessage

from ldp.graph import FxnOp, OpResult, PromptOp, compute_graph
from ldp.llms import prepend_sys

from .llm_call import ParsedLLMCallModule

_DEFAULT_PROMPT_TEMPLATE = textwrap.dedent(
    """    Answer the following questions as best you can. You have access to the following tools:

    {{tools}}

    Use the following format:

    {fields}
    ... (this {fields_description} can repeat N times)

    Example:

    {example}"""
)
REACT_DEFAULT_PROMPT_TEMPLATE = _DEFAULT_PROMPT_TEMPLATE.format(
    fields=(
        "Thought: you should always think about what to do"
        "\nAction: the action to take, should be one of [{tool_names}]"
        "\nAction Input: comma separated list of inputs to action as python tuple"
        "\nObservation: the result of the action"
    ),
    fields_description="Thought/Action/Action Input/Observation",
    example=(
        "Thought: I need to use the get_weather tool"
        "\nAction: get_weather"
        '\nAction Input: "New York", 7'
        "\nObservation: The 7 day forecast for New York is [...]"
    ),
)
ACT_DEFAULT_PROMPT_TEMPLATE = _DEFAULT_PROMPT_TEMPLATE.format(
    fields=(
        "Action: the action to take, should be one of [{tool_names}]"
        "\nAction Input: comma separated list of inputs to action as python tuple"
        "\nObservation: the result of the action"
    ),
    fields_description="Action/Action Input/Observation",
    example=(
        "Action: get_weather"
        '\nAction Input: "New York", 7'
        "\nObservation: The 7 day forecast for New York is [...]"
    ),
)


def parse_message(m: Message, tools: list[Tool]) -> ToolRequestMessage:  # noqa: C901
    """
    Parse an Act or ReAct Message into a ToolRequestMessage.

    Args:
        m: Input raw message.
        tools: Tools used to confirm a valid tool selection

    Returns:
        Parsed ToolRequestMessage.
    """
    if not m.content:
        raise MalformedMessageError(
            f"{EMPTY_CONTENT_BASE_MSG} of type {type(m).__name__}."
        )

    message_content = m.content
    # strip (and overwrite) up to end of action input
    loc = message_content.find("Action Input:")
    if loc != -1:
        loc = message_content.find("\n", loc)
        message_content = message_content[: loc if loc > 0 else None]
    # we need to override the message too - don't want the model to hallucinate
    m.content = message_content
    
    print("Received back!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n" + m.content)

    action_args: tuple[Any, ...] = ()
    # https://regex101.com/r/qmqZ7Z/1
    action_input = re.search(r"Input:[ \t]*([ \S]*)", m.content)
    # only parse if it takes arguments
    if action_input and action_input.group(1).strip():
        input_str = action_input.group(1).strip()
        # if it has commas and no quotes, it's almost certainly a tuple without
        # parentheses, so we add them
        if "," in input_str and not (
            input_str.startswith("(") and input_str.endswith(")")
        ):
            input_str = f"({input_str})"
        try:
            if input_str.startswith("(") and input_str.endswith(")"):
                # Handle tuples and quoted strings inside
                if '"' not in input_str and "'" not in input_str:
                    # Add quotes around each element within parentheses if they are not already quoted
                    # and if they are not purely numbers. There may exist a killer regex for this
                    # but I am a simple man

                    # just catches things like "1.1".isnumeric() == False
                    # so we can't just use isnumeric
                    def is_number(s: str) -> bool:
                        try:
                            float(s)
                        except ValueError:
                            return False
                        return True

                    input_str = ", ".join(
                        f'"{e.strip()}"' if not is_number(e) else str(e)
                        for e in input_str.strip("()").split(",")
                        if e.strip()
                    )
                    input_str = f"({input_str})"
                eval_result = ast.literal_eval(input_str)
                action_args = (
                    (eval_result,)
                    if not isinstance(eval_result, tuple)
                    else eval_result
                )
            else:
                # Convert to int or float if possible
                try:
                    action_args = (ast.literal_eval(input_str),)
                except (ValueError, SyntaxError):
                    action_args = (input_str,)
        except Exception as exc:
            raise MalformedMessageError(
                f"Action Input {input_str} could not be parsed."
            ) from exc

        if len(action_args) == 1 and isinstance(action_args[0], tuple):
            action_args = action_args[0]

    action = re.search(r"Action:[ \t]*(\S*)", m.content)
    if not action:
        raise MalformedMessageError("Action not emitted.")
    if "Observation:" in m.content:
        raise MalformedMessageError("Observation found in message content and not expected.")
    tool_name = action.group(1).strip()
    # have to match up name to tool to line up args in order
    try:
        tool = next(t for t in tools if t.info.name == tool_name)
    except StopIteration as exc:
        raise MalformedMessageError(f"Tool {tool_name} not found in tools.") from exc
    if len(action_args) < len(tool.info.parameters.required):
        raise MalformedMessageError(
            f"Action Input {action_args!r} shorter than {tool.info.name!r} tool's"
            " parameters."
        )

    # Anecdotally we've observed thought also often captures the action
    # NOTE: for Act agents there is no Thought, so the regex will return None
    thought = re.search(r"Thought:[ \t]*(.*)", m.content)
    return ToolRequestMessage(
        content=thought.group(1) if thought else None,
        tool_calls=[ToolCall.from_tool(tool, *action_args)],
    )


class ToolDescriptionMethods(StrEnum):
    """Possible methods of describing the tools."""

    STR = "describe_str"
    XML = "describe_xml"
    JSON = "describe_json"

    def get_prompt_prefix(self) -> str:
        """Get the prefix to put in front of the prompt."""
        if self == self.STR:
            return ""
        if self == self.JSON:
            return "Tools are specified with a JSON schema."
        return "Tools are specified with an XML schema."


class ReActModule:
    """An Act or ReAct module built to work with chat models.

    Paper: https://arxiv.org/abs/2210.03629

    The ReAct style is like so, and note Act style has no 'Thought: ' entries:
    System:
        Answer the following questions as best you can. You have access to the following tools:

            {tools}

            Use the following format:

            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
    User:
        {questions}
    Assistant:
        Thought:
        Action:
        Action Input:
    User:
        Observation:
    Assistant:
        Thought:
        Action:
        Action Input:
    ...
    """

    @staticmethod
    def parse_message(m: Message, tools: list[Tool]) -> ToolRequestMessage:
        return parse_message(m, tools)

    async def _create_system_prompt(self, tools: list[Tool]) -> OpResult[str]:
        tool_info = "\n".join([
            getattr(t.info, self._tool_description_method)() for t in tools
        ])
        if prefix := self._tool_description_method.get_prompt_prefix():
            tool_info = f"{prefix}\n{tool_info}"
        tool_names = ", ".join([t.info.name for t in tools])
        return await self.prompt_op(
            schema_type=self._tool_description_method.value,
            tools=tool_info.strip(),
            tool_names=tool_names,
        )

    def __init__(
        self,
        llm_model: dict[str, Any],
        sys_prompt: str = REACT_DEFAULT_PROMPT_TEMPLATE,
        tool_description_method: ToolDescriptionMethods = ToolDescriptionMethods.STR,
    ):
        self.prompt_op = PromptOp(sys_prompt)
        self._tool_description_method = tool_description_method
        llm_model["stop"] = ["Observation:"]
        self.package_msg_op = FxnOp(prepend_sys)
        self.tool_select_module = ParsedLLMCallModule[ToolRequestMessage](
            llm_model=llm_model, parser=self.parse_message
        )

    @compute_graph()
    async def __call__(
        self, messages: Iterable[Message], tools: list[Tool]
    ) -> tuple[OpResult[ToolRequestMessage], Message]:
        packaged_msgs = await self.package_msg_op(
            messages, sys_content=await self._create_system_prompt(tools)
        )
        return await self.tool_select_module(packaged_msgs, tools=tools)  # type: ignore[arg-type]
