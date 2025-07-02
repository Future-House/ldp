import re
import xml.etree.ElementTree as ET

from aviary.core import Message, Tool, ToolCall, ToolCallFunction, ToolRequestMessage

from .simple_agent import NoToolsSimpleAgent


def _parse_argument_value(value: str) -> str | int | float | bool:
    """Parse a string value into the appropriate type.

    Args:
        value: The string value to parse

    Returns:
        The parsed value with appropriate type
    """
    value = value.strip()

    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.isdigit():
        return int(value)
    return float(value) if value.replace(".", "", 1).isdigit() else value


def _extract_arguments(args_elem) -> dict:
    """Extract arguments from the arguments XML element.

    Args:
        args_elem: The arguments XML element

    Returns:
        Dictionary of argument names to parsed values
    """
    arguments: dict[str, str | int | float | bool] = {}
    if args_elem is None:
        return arguments

    for arg_elem in args_elem:
        if arg_elem.tag and arg_elem.text is not None:
            arguments[arg_elem.tag] = _parse_argument_value(arg_elem.text)

    return arguments


def _parse_single_tool_call(match: str) -> ToolCall | None:
    """Parse a single tool call from XML string.

    Args:
        match: The XML content of a single tool call

    Returns:
        ToolCall object if parsing succeeds, None otherwise
    """
    try:
        xml_str = f"<tool_call>{match}</tool_call>"
        root = ET.fromstring(xml_str)  # noqa: S314

        # Extract ID from attributes if present, otherwise generate one
        tool_id = root.get("id", ToolCall.generate_id())

        # Extract function name
        name_elem = root.find("name")
        if name_elem is None or name_elem.text is None:
            return None
        function_name = name_elem.text.strip()

        # Extract arguments
        args_elem = root.find("arguments")
        arguments = _extract_arguments(args_elem)

        # Create ToolCall
        return ToolCall(
            id=tool_id,
            function=ToolCallFunction(name=function_name, arguments=arguments),
        )

    except ET.ParseError:
        return None


def parse_xml_tool_calls(text: str) -> list[ToolCall]:
    """Parse XML tool calls from text response.

    Expected format:
    <tool_call>
      <name>function_name</name>
      <arguments>
        <param1>value1</param1>
        <param2>value2</param2>
      </arguments>
    </tool_call>

    Args:
        text: The raw text response containing XML tool calls

    Returns:
        List of ToolCall objects parsed from the XML
    """
    # Find all tool_call blocks in the text
    tool_call_pattern = r"<tool_call[^>]*>(.*?)</tool_call>"
    matches = re.findall(tool_call_pattern, text, re.DOTALL)

    # Parse each match and filter out None results
    tool_calls = []
    for match in matches:
        tool_call = _parse_single_tool_call(match)
        if tool_call is not None:
            tool_calls.append(tool_call)

    return tool_calls


def generate_xml_tool_instructions(tools: list[Tool]) -> str:
    """Generate system prompt instructions for using tools in XML format.

    Args:
        tools: List of available tools

    Returns:
        System prompt instruction text for XML tool usage
    """
    if not tools:
        return ""

    instructions = [
        "You have access to the following tools. When you need to use a tool, "
        "format your response using XML tags as shown below.",
        "",
        "Available tools:",
    ]

    # Add tool descriptions
    for tool in tools:
        instructions.append(f"- {tool.info.name}: {tool.info.description}")
        if tool.info.parameters and tool.info.parameters.properties:
            instructions.append("  Parameters:")
            for param_name, param_info in tool.info.parameters.properties.items():
                param_type = param_info.get("type", "unknown")
                param_desc = param_info.get("description", "No description")
                required = param_name in (tool.info.parameters.required or [])
                req_str = " (required)" if required else " (optional)"
                instructions.append(
                    f"    - {param_name} ({param_type}){req_str}: {param_desc}"
                )
        instructions.append("")

    instructions.extend([
        "To use a tool, format your response like this:",
        "",
        '<tool_call id="unique_id">',
        "  <name>tool_name</name>",
        "  <arguments>",
        "    <parameter_name>parameter_value</parameter_name>",
        "    <another_param>another_value</another_param>",
        "  </arguments>",
        "</tool_call>",
        "",
        "You can make multiple tool calls by including multiple <tool_call> blocks.",
        "You may provide a unique id for each tool call, but it is not required.",
        "Include any reasoning or explanation outside of the tool_call blocks.",
    ])

    return "\n".join(instructions)


def xml_tool_parser(message: Message) -> ToolRequestMessage:
    """Parse a plain text message into a ToolRequestMessage with XML tool calls.

    Args:
        message: The plain text message from the LLM

    Returns:
        ToolRequestMessage with parsed tool calls
    """
    content = message.content or ""
    tool_calls = parse_xml_tool_calls(content)

    return ToolRequestMessage(content=content, tool_calls=tool_calls)


class XMLToolAgent(NoToolsSimpleAgent):
    """Agent that uses XML format for tool calls instead of JSON."""

    def __init__(self, **kwargs):
        """Initialize XMLToolAgent.

        Args:
            **kwargs: Keyword arguments passed to NoToolsSimpleAgent
        """
        super().__init__(cast_tool_request=xml_tool_parser, **kwargs)

    async def init_state(self, tools: list[Tool]):
        """Initialize agent state with XML tool instructions.

        Args:
            tools: List of available tools

        Returns:
            Initialized SimpleAgentState
        """
        state = await super().init_state(tools)

        if xml_instructions := generate_xml_tool_instructions(tools):
            # prepend system prompt with xml instructions
            state.messages.insert(0, Message(content=xml_instructions, role="system"))

        return state


def extract_code_blocks(text: str) -> list[str]:
    """Extract all code blocks from the given text.

    Args:
        text (str): The text to extract the code blocks from.

    Returns:
        list[str]: A list of all code blocks found in the text.
    """
    pattern = r"```(?:\w+\n)?([\s\S]*?)```"
    matches = re.findall(pattern, text)

    return [match.strip() for match in matches]


def extract_cells(content: str) -> list[tuple[int, str]]:
    cell_matches = re.findall(
        r"Cell (\d+)(?:\s*\([^)]*\))?\s*:\s*\n```python\n(.*?)\n```",
        content or "",
        re.DOTALL,
    )
    return [(int(cell_idx), code_block) for cell_idx, code_block in cell_matches]


def parse_simpler_notebook_agent_action(message: Message) -> ToolRequestMessage:
    """Parse the action from the message.

    Cell 0:
    ```python
    print("Hello, world!")
    ```

    Cell 1:
    ```python
    import pandas as pd

    df = pd.read_csv("datasets/brain_size_data.csv")
    print(df.head())
    ```

    <answer>
    Your final answer here
    </answer>

    """
    # Check for <answer></answer> tag using regex
    # Make sure we support multi-line answers
    if answer_match := re.search(
        r"<answer>(.*?)</answer>", message.content or "", re.DOTALL
    ):
        return ToolRequestMessage(
            content=answer_match[1],
            tool_calls=[ToolCall.from_name("submit_answer", answer=answer_match[1])],
        )

    if cells := extract_cells(message.content or ""):
        tool_calls: list[ToolCall] = []
        tool_calls.extend(
            ToolCall.from_name("edit_cell", contents=code_block, idx=cell_idx)
            for cell_idx, code_block in cells
        )
        return ToolRequestMessage(
            content=message.content,
            tool_calls=tool_calls,
        )
    # If a code block is found without a cell index, we assume it is the first cell
    code_blocks = extract_code_blocks(message.content or "")
    if code_blocks:
        return ToolRequestMessage(
            content=message.content,
            tool_calls=[
                ToolCall.from_name("edit_cell", contents=code_blocks[0], idx=None)
            ],
        )
    return ToolRequestMessage(content=message.content, tool_calls=[])


class SimplerNotebookAgent(NoToolsSimpleAgent):
    """Agent that uses a simpler notebook format for tool calls."""

    def __init__(self, **kwargs):
        """Initialize SimplerNotebookAgent.

        Args:
            **kwargs: Keyword arguments passed to NoToolsSimpleAgent
        """
        super().__init__(
            cast_tool_request=parse_simpler_notebook_agent_action, **kwargs
        )
