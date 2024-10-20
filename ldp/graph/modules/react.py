import textwrap
from collections.abc import Iterable
from enum import StrEnum
from typing import Any, cast

from aviary.message import Message
from aviary.tools import Tool, ToolRequestMessage

from ldp.graph import FxnOp, LLMCallOp, OpResult, PromptOp, compute_graph
from ldp.llms import append_to_messages, prepend_sys

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


def generate_tool_selection_prompt(react_message) -> Message:
    reasoning_template = textwrap.dedent("""
        {reasoning}. Based on this reasoning, let's select the appropriate tool!
    """)

    # Prepare the content using the template and provided arguments
    content = reasoning_template.format(reasoning=react_message.content or "")

    # Return the formatted Message
    return Message(content=content, role="assistant")


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
        self.tool_selection_msg_op = FxnOp(generate_tool_selection_prompt)
        self._tool_description_method = tool_description_method
        llm_model["stop"] = ["Observation:"]
        self.llm_config = llm_model
        self.package_msg_op = FxnOp(prepend_sys)
        self.append_msg_op = FxnOp(append_to_messages)
        self.llm_call_op = LLMCallOp()

    @compute_graph()
    async def __call__(
        self, messages: Iterable[Message], tools: list[Tool]
    ) -> tuple[OpResult[ToolRequestMessage], Message]:
        packaged_msgs = await self.package_msg_op(
            messages, sys_content=await self._create_system_prompt(tools)
        )
        # Ask the LLM to do the reasoning
        reasoning_msg = await self.llm_call_op(
            self.llm_config,
            msgs=packaged_msgs,
            tools=tools,
            tool_choice="none",
        )
        # Add the reasoning to messages. Generate the tool selection prompt
        packaged_msgs_with_reasoning = await self.package_msg_op(
            await self.append_msg_op(
                messages, await self.tool_selection_msg_op(reasoning_msg)
            ),
            sys_content=await self._create_system_prompt(tools),
        )
        # Ask the LLM to select the tool
        tool_selection_msg = await self.llm_call_op(
            self.llm_config, msgs=packaged_msgs_with_reasoning, tools=tools
        )
        return cast(
            OpResult[ToolRequestMessage], tool_selection_msg
        ), reasoning_msg.value
