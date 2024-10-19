from unittest.mock import Mock, patch

import pytest
from aviary.env import DummyEnv
from aviary.message import Message
from aviary.tools import Tool, ToolRequestMessage

from ldp.agent import ReActAgent
from ldp.graph import OpResult
from ldp.graph.modules import (
    ReActModule,
    ReflectModule,
    ReflectModuleConfig,
)

from . import CILLMModelNames


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_reflect_module() -> None:
    config = ReflectModuleConfig(
        llm_model={"model": CILLMModelNames.ANTHROPIC.value, "temperature": 0}
    )  # Lower temperature for more deterministic responses
    reflect_module = ReflectModule(config)
    context = "I am happy. How do I feel?"
    response = "You are sad."
    result = await reflect_module(context, response)
    assert isinstance(result, OpResult)
    assert len(result.value) > 0
    assert result.value != response
    # Check both emotions to work around LLM not responding with "happy"
    # For example: "It sounds like you are feeling joyful."
    assert "happy" in result.value or "sad" not in result.value


@pytest.fixture(name="mock_tools")
def fixture_mock_tools() -> list[Tool]:
    return [Mock(spec=Tool, name=f"Tool{i}") for i in range(3)]


class TestReActModule:
    @pytest.mark.asyncio
    async def test_templating(self, dummy_env: DummyEnv) -> None:
        obs, tools = await dummy_env.reset()
        module = ReActModule(ReActAgent.model_fields["llm_model"].default)
        with patch(
            "ldp.graph.common_ops.LLMCallOp.forward",
            return_value=ToolRequestMessage(
                role="assistant",
                content=f"Action: {tools[0].info.name}\nAction Input: stub",
            ),
        ) as mock_forward:
            await module(obs, tools=tools)
        mock_forward.assert_awaited_once()
        assert mock_forward.await_args
        assert mock_forward.await_args[1]["msgs"][0] == Message(
            role="system",
            content=(
                "Answer the following questions as best you can. You have access to the"
                " following tools:\n\nNAME: print_story\n\nSYNOPSIS:\n   "
                " print_story(string story)\n\nDESCRIPTION:\n    Print a"
                " story.\n\nPARAMETERS:\n    story (string): Story to print.\n\nNAME:"
                " cast_float\n\nSYNOPSIS:\n    cast_float(string"
                " x)\n\nDESCRIPTION:\n    Cast the input argument x to a"
                " float.\n\nPARAMETERS:\n    x (string): No description"
                " provided.\n\nNAME: cast_int\n\nSYNOPSIS:\n   "
                " cast_int(number x)\n\nDESCRIPTION:\n    Cast the input argument x to"
                " an integer.\n\nPARAMETERS:\n    x (number): No description"
                " provided.\n\nUse the following format:\n\nThought: you should"
                " always think about what to do\nAction: the action to take, should be"
                " one of [print_story, cast_float, cast_int]\nAction Input: comma"
                " separated list of inputs to action as python tuple\nObservation: the"
                " result of the action\n... (this Thought/Action/Action"
                " Input/Observation can repeat N times)\n\nExample:\n\nThought: I need"
                ' to use the get_weather tool\nAction: get_weather\nAction Input: "New'
                ' York", 7\nObservation: The 7 day forecast for New York is [...]'
            ),
        )
