import ast
import re
import textwrap
from collections.abc import Iterable
from enum import StrEnum
from typing import Any, cast

from aviary.core import (
    MalformedMessageError,
    Message,
    Messages,
    Tool,
    ToolCall,
    ToolRequestMessage,
)
from aviary.message import EMPTY_CONTENT_BASE_MSG

from ldp.graph import FxnOp, LLMCallOp, OpResult, PromptOp, compute_graph
from ldp.llms import prepend_sys

from .llm_call import ParsedLLMCallModule


def clean_llm_output(
    content: str, prefix: str | None = None, truncate_at: str | None = None
) -> str:
    """
    Clean LLM output by removing prefix and truncating at specified marker.

    Args:
        content: The raw LLM output content
        prefix: Optional prefix to remove (e.g., "Plan:", "Thought:")
        truncate_at: Optional marker to truncate at (e.g., "Action:", "Thought:")

    Returns:
        Cleaned content string
    """
    if not content:
        return content

    cleaned_content = content

    # Remove prefix if present
    if prefix and cleaned_content.startswith(prefix):
        cleaned_content = cleaned_content[len(prefix) :].strip()

    # Truncate at marker if present
    if truncate_at and truncate_at in cleaned_content:
        cleaned_content = cleaned_content.split(truncate_at)[0].strip()

    return cleaned_content


# These prompts are meant to be used with ReActModuleSinglePrompt
_DEFAULT_SINGLE_PROMPT_TEMPLATE = textwrap.dedent(
    """    Answer the following questions as best you can. You have access to the following tools:

    {{tools}}

    Use the following format:

    {fields}
    ... (this {fields_description} can repeat N times)

    Example:

    {example}"""
)
REACT_DEFAULT_SINGLE_PROMPT_TEMPLATE = _DEFAULT_SINGLE_PROMPT_TEMPLATE.format(
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
ACT_DEFAULT_SINGLE_PROMPT_TEMPLATE = _DEFAULT_SINGLE_PROMPT_TEMPLATE.format(
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


# And these with ReActModule
_DEFAULT_PROMPT_TEMPLATE = textwrap.dedent(
    """    Answer the following questions as best you can, using the provided tools.

    Use the following format:

    {fields}
    ... (this {fields_description} can repeat N times)

    Example:

    {example}"""
)
REACT_DEFAULT_PROMPT_TEMPLATE = _DEFAULT_PROMPT_TEMPLATE.format(
    fields=(
        "Thought: you should always think about what to do"
        "\nAction: the action to take,"
        " should be one of the provided tools with necessary arguments"
        "\nObservation: the result of the action"
    ),
    fields_description="Thought/Action/Observation",
    example=(
        "Thought: I need to use the get_weather tool"
        '\nAction: get_weather("New York", 7)'
        "\nObservation: The 7 day forecast for New York is [...]"
    ),
)
ACT_DEFAULT_PROMPT_TEMPLATE = _DEFAULT_PROMPT_TEMPLATE.format(
    fields=(
        "Action: the action to take,"
        " should be one of the provided tools with necessary arguments"
        "\nObservation: the result of the action"
    ),
    fields_description="Action/Observation",
    example=(
        'Action: get_weather("New York", 7)'
        "\nObservation: The 7 day forecast for New York is [...]"
    ),
)

# Planning mode prompt template
REACT_PLANNING_PROMPT_TEMPLATE = _DEFAULT_PROMPT_TEMPLATE.format(
    fields=(
        "Critic: Assess whether the previous step of the trajectory has successfully completed the last step of the "
        "plan."
        "\nPlan: Give an updated plan as a checklist with [ ] for incomplete and [x] for completed steps. "
        "Each step should be ~3 sentences long. Below each step, include a list of criteria for what counts as "
        "satisfying that particular step."
        "\nThought: Reason about the immediate next step you're about to take"
        "\nAction: the action to take, should be one of the provided tools with necessary arguments"
        "\nObservation: the result of the action"
    ),
    fields_description="Critic/Plan/Thought/Action/Observation",
    example=(
        "Critic: This is the first step, so not applicable."
        "\nPlan: Updated plan:\n[ ] Get weather information for New York. This involves calling the weather API with "
        "the correct parameters. "
        "We need to retrieve a comprehensive 7-day forecast that includes temperature, conditions, and any relevant "
        "weather alerts.\n"
        "  - Successfully called get_weather function with correct parameters\n"
        "  - Received valid weather data response\n"
        "  - Data includes temperature and forecast information\n"
        "[ ] Format the response appropriately. Take the raw weather data and present it in a clear, readable format "
        "for the user. "
        "The response should be well-structured and easy to understand.\n"
        "  - Weather data is organized in a logical format\n"
        "  - Information is presented clearly and concisely\n"
        "  - Response addresses the user's original question"
        "\nThought: I need to start by getting the weather information for New York using the get_weather tool"
        '\nAction: get_weather("New York", 7)'
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
    tool_name = action.group(1).strip()
    # have to match up name to tool to line up args in order
    try:
        tool = next(t for t in tools if t.info.name == tool_name)
    except StopIteration as exc:
        raise MalformedMessageError(f"Tool {tool_name} not found in tools.") from exc
    required_parameters = tool.info.parameters.required if tool.info.parameters else []
    if len(action_args) < len(required_parameters):
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


class ReActModuleSinglePrompt:
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
        sys_prompt: str = REACT_DEFAULT_SINGLE_PROMPT_TEMPLATE,
        tool_description_method: ToolDescriptionMethods = ToolDescriptionMethods.STR,
    ):
        self.prompt_op = PromptOp(sys_prompt)
        self._tool_description_method = tool_description_method
        llm_model["stop"] = ["Observation:"]
        self.package_msg_op = FxnOp(prepend_sys)
        self.tool_select_module = ParsedLLMCallModule[ToolRequestMessage](
            llm_model=llm_model, parser=self.parse_message
        )

    @property
    def llm_call_op(self) -> LLMCallOp:
        return self.tool_select_module.llm_call_op

    @compute_graph()
    async def __call__(
        self, messages: Iterable[Message], tools: list[Tool]
    ) -> tuple[OpResult[ToolRequestMessage], list[Message]]:
        packaged_msgs = await self.package_msg_op(
            messages, sys_content=await self._create_system_prompt(tools)
        )
        final_result, react_message = await self.tool_select_module(
            packaged_msgs,  # type: ignore[arg-type]
            tools=tools,
        )
        return final_result, [react_message]


def postprocess_and_concat_resoning_msg(
    msgs: Iterable[Message], react_message: Message
) -> Messages:
    reasoning = (react_message.content or "").removeprefix("Thought: ")
    return [
        *msgs,
        Message(
            content=(
                f"Thought: {reasoning}."
                " Based on this reasoning, let's select the appropriate tool!"
                "\nAction: "
            ),
            # Role is 'assistant' here (normally 'user') since we use the model's reasoning to ask for an action.
            role="assistant",
            info={"is_thought": True},
        ),
        # We interleave a user message as required by Anthropic's API
        Message(content="Continue..."),
    ]


class ReActModule(ReActModuleSinglePrompt):
    def __init__(
        self,
        llm_model: dict[str, Any],
        sys_prompt: str = REACT_DEFAULT_PROMPT_TEMPLATE,
        tool_description_method: ToolDescriptionMethods = ToolDescriptionMethods.STR,
    ):
        self._tool_description_method = tool_description_method
        llm_model["stop"] = ["Observation:", "Action:"]
        self.llm_config = llm_model
        self._llm_call_op = LLMCallOp()
        self.prompt_op = PromptOp(sys_prompt)
        self.package_msg_op = FxnOp(prepend_sys)
        self.postprocess_reasoning_msg_op = FxnOp(postprocess_and_concat_resoning_msg)

    async def _create_system_prompt(self, tools: list[Tool]) -> OpResult[str]:
        raise NotImplementedError(
            "ReActModule does not implement _create_system_prompt, "
            "since tool descriptions are passed to the API directly "
            "instead of via prompt. Use self.prompt_op instead."
        )

    @property
    def llm_call_op(self) -> LLMCallOp:
        return self._llm_call_op

    @compute_graph()
    async def __call__(
        self, messages: Iterable[Message], tools: list[Tool]
    ) -> tuple[OpResult[ToolRequestMessage], Messages]:
        sys_prompt = await self.prompt_op()

        packaged_msgs = await self.package_msg_op(messages, sys_content=sys_prompt)
        # Ask the LLM to do the reasoning
        reasoning_msg = await self.llm_call_op(
            self.llm_config,
            msgs=packaged_msgs,
            tools=tools,
            tool_choice="none",  # Reasoning shouldn't pick a tool
        )
        # Add the reasoning to messages. Generate the tool selection prompt
        packaged_msgs_with_reasoning = await self.package_msg_op(
            await self.postprocess_reasoning_msg_op(messages, reasoning_msg),
            sys_content=sys_prompt,
        )
        # Ask the LLM to select the tool
        tool_selection_msg = await self.llm_call_op(
            self.llm_config, msgs=packaged_msgs_with_reasoning, tools=tools
        )
        return cast("OpResult[ToolRequestMessage]", tool_selection_msg), [
            # We return the 3 new messages: reasoning (assistant) message,
            # the "continue..." (user) message from user,
            # and tool selection (assistant) message
            *packaged_msgs_with_reasoning.value[-2:],
            tool_selection_msg.value,
        ]


class ReActPlanningModule(ReActModule):
    """A planning variant of ReActModule that makes three separate LLM calls for structured reasoning."""

    def __init__(
        self,
        llm_model: dict[str, Any],
        sys_prompt: str = REACT_PLANNING_PROMPT_TEMPLATE,
        tool_description_method: ToolDescriptionMethods = ToolDescriptionMethods.STR,
    ):
        self._tool_description_method = tool_description_method
        llm_model["stop"] = ["Observation:", "Action:"]
        self.llm_config = llm_model
        self._llm_call_op = LLMCallOp()
        self.prompt_op = PromptOp(sys_prompt)
        self.package_msg_op = FxnOp(prepend_sys)

        # Create prompt ops for the three components
        self.critic_prompt_op = PromptOp(
            "Output ONLY a critic assessment. Assess whether the latest step of the trajectory has successfully "
            "completed the latest step of the plan or not. Be critical and thorough to catch mistakes made in the "
            "execution of the plan, even if minor. Use common sense: think of any mistakes the agent might have made "
            "(not only in the code) but also in their reasoning or analysis process. Do not output plan or thought."
        )
        self.plan_prompt_op = PromptOp(
            "Output ONLY an updated plan. Give an updated plan as a checklist with [ ] for incomplete and [x] for "
            "completed steps, where each step is ~3 sentences long. Below each step, include a list of criteria for "
            "what counts as satisfying that particular step. Do not output critic or thought."
        )
        self.thought_prompt_op = PromptOp(
            "Output ONLY a thought. Reason concretely about the immediate next step you're about to take, rather than "
            "presenting several options about what to do. Make sure you are clearly addressing the next step of the "
            "plan and resolving any criticism where applicable.  Do not output critic or plan."
        )

    @property
    def llm_call_op(self) -> LLMCallOp:
        return self._llm_call_op

    async def _execute_step(
        self,
        messages: list[Message],
        sys_prompt: OpResult[str],
        instruction_prompt_op: PromptOp,
        tools: list[Tool],
        prefix: str | None = None,
        truncate_at: str | None = None,
    ) -> Message:
        """
        Helper method to execute a single step (Critic, Plan, or Thought).

        Args:
            messages: Current messages
            sys_prompt: System prompt
            instruction_prompt_op: Prompt op for the step instruction
            tools: Available tools
            prefix: Optional prefix to clean from output
            truncate_at: Optional marker to truncate output at

        Returns:
            The processed message from the LLM
        """
        instruction = await instruction_prompt_op()
        packaged_msgs = await self.package_msg_op(messages, sys_content=sys_prompt)
        step_msgs = [
            *packaged_msgs.value,
            Message(content=instruction.value),
        ]
        step_result = await self.llm_call_op(
            self.llm_config,
            msgs=step_msgs,
            tools=tools,
            tool_choice="none",
        )
        step_msg = step_result.value

        # Clean up output if prefix or truncate_at are provided
        if step_msg.content and (prefix is not None or truncate_at is not None):
            step_msg.content = clean_llm_output(
                step_msg.content, prefix=prefix, truncate_at=truncate_at
            )

        return step_msg

    @compute_graph()
    async def __call__(
        self, messages: Iterable[Message], tools: list[Tool]
    ) -> tuple[OpResult[ToolRequestMessage], Messages]:
        sys_prompt = await self.prompt_op()
        messages = list(messages)

        # Check if this is the first step (no previous tool responses)
        is_first_step = not any(
            hasattr(msg, "tool_calls") and msg.tool_calls for msg in messages
        )

        # Step 1: Critic - assess previous step (skip on first step)
        if is_first_step:
            critic_msg = None
            current_messages = messages
        else:
            critic_msg = await self._execute_step(
                messages, sys_prompt, self.critic_prompt_op, tools, "Critic:", "Plan:"
            )
            current_messages = [*messages, critic_msg]

        # Step 2: Plan - generate updated plan
        plan_msg = await self._execute_step(
            current_messages, sys_prompt, self.plan_prompt_op, tools, "Plan:", "Thought:"
        )

        # Step 3: Thought - reason about immediate next step
        thought_msg = await self._execute_step(
            [*current_messages, plan_msg], sys_prompt, self.thought_prompt_op, tools, "Thought:", "Action:"
        )

        # Combine all reasoning into a single message for tool selection
        if is_first_step:
            combined_reasoning = (
                f"Plan: {plan_msg.content or ''}\n\n"
                f"Thought: {thought_msg.content or ''}"
            )
        else:
            assert critic_msg is not None
            combined_reasoning = (
                f"Critic: {critic_msg.content or ''}\n\n"
                f"Plan: {plan_msg.content or ''}\n\n"
                f"Thought: {thought_msg.content or ''}"
            )

        # Step 4: Tool selection based on combined reasoning
        tool_selection_prompt = (
            f"{combined_reasoning}."
            " Based on this reasoning, let's select the appropriate tool!"
            "\n\nAction: "
        )

        # Create the new messages that will be added
        new_messages = [
            Message(content=tool_selection_prompt, role="assistant", info={"is_thought": True}),
            Message(content="Continue..."),
        ]

        packaged_msgs_final = await self.package_msg_op(
            [*messages, *new_messages],
            sys_content=sys_prompt,
        )

        tool_selection_msg = await self.llm_call_op(
            self.llm_config, msgs=packaged_msgs_final, tools=tools
        )

        return cast("OpResult[ToolRequestMessage]", tool_selection_msg), [
            *new_messages,
            tool_selection_msg.value,
        ]
