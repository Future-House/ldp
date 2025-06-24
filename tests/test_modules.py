from unittest.mock import Mock, patch

import pytest
from aviary.core import DummyEnv, Message, Tool, ToolRequestMessage
from lmi import CommonLLMNames
from pytest_subtests import SubTests

from ldp.agent import ReActAgent
from ldp.graph import OpResult
from ldp.graph.modules import (
    ReActModule,
    ReActModuleSinglePrompt,
    ReflectModule,
    ReflectModuleConfig,
    parse_message,
)


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_reflect_module() -> None:
    config = ReflectModuleConfig(
        llm_model={
            "name": CommonLLMNames.ANTHROPIC_TEST.value,
            "temperature": 0,  # Lower temperature for more deterministic responses
        }
    )
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
    @pytest.mark.parametrize("single_prompt", [True, False])
    async def test_templating(self, dummy_env: DummyEnv, single_prompt: bool) -> None:
        obs, tools = await dummy_env.reset()
        if single_prompt:
            module = ReActModuleSinglePrompt(
                ReActAgent.model_fields["llm_model"].default
            )
        else:
            module = ReActModule(ReActAgent.model_fields["llm_model"].default)
        with patch(
            "ldp.graph.common_ops.LLMCallOp.forward",
            return_value=ToolRequestMessage(
                role="assistant",
                content=f"Action: {tools[0].info.name}\nAction Input: stub",
            ),
        ) as mock_forward:
            await module(obs, tools=tools)
        if single_prompt:
            mock_forward.assert_awaited_once()
        else:
            assert mock_forward.await_count == 2
        assert mock_forward.await_args

        if single_prompt:
            expected_message = Message(
                role="system",
                content=(
                    "Answer the following questions as best you can. You have access to"
                    " the following tools:\n\nNAME: print_story\n\nSYNOPSIS:\n   "
                    " print_story(string story)\n\nDESCRIPTION:\n    Print a"
                    " story.\n\nPARAMETERS:\n    story (string): Story to"
                    " print.\n\nNAME: cast_float\n\nSYNOPSIS:\n    cast_float(string"
                    " x)\n\nDESCRIPTION:\n    Cast the input argument x to a"
                    " float.\n\nPARAMETERS:\n    x (string): No description"
                    " provided.\n\nNAME: cast_int\n\nSYNOPSIS:\n    cast_int(number"
                    " x)\n\nDESCRIPTION:\n    Cast the input argument x to an"
                    " integer.\n\nPARAMETERS:\n    x (number): No description"
                    " provided.\n\nNAME: get_random_int\n\nSYNOPSIS:\n   "
                    " get_random_int()\n\nDESCRIPTION:\n    Get a random integer in 1"
                    " to 10.\n\nPARAMETERS:\n\nUse the following format:\n\nThought:"
                    " you should always think about what to do\nAction: the action to"
                    " take, should be one of [print_story, cast_float, cast_int,"
                    " get_random_int]\nAction Input: comma separated list of inputs to"
                    " action as python tuple\nObservation: the result of the"
                    " action\n... (this Thought/Action/Action Input/Observation can"
                    " repeat N times)\n\nExample:\n\nThought: I need to use the"
                    ' get_weather tool\nAction: get_weather\nAction Input: "New York",'
                    " 7\nObservation: The 7 day forecast for New York is [...]"
                ),
            )
        else:
            expected_message = Message(
                role="system",
                content=(
                    "Answer the following questions as best you can, using the provided"
                    " tools.\n\nUse the following format:\n\nThought: you should always"
                    " think about what to do\nAction: the action to take, should be one"
                    " of the provided tools with necessary arguments\nObservation: the"
                    " result of the action\n... (this Thought/Action/Observation can"
                    " repeat N times)\n\nExample:\n\nThought: I need to use the"
                    ' get_weather tool\nAction: get_weather("New York",'
                    " 7)\nObservation: The 7 day forecast for New York is [...]"
                ),
            )

        assert mock_forward.await_args[1]["msgs"][0] == expected_message

    @pytest.mark.asyncio
    async def test_react_parse(self, subtests: SubTests) -> None:  # noqa: PLR0915, C901
        def zero_arg() -> int:
            """A test function."""
            return 0

        def one_arg(x: int) -> int:
            """A test function.

            Args:
                x: x
            """
            return x

        def one_string_arg(x: str) -> str:
            """A test function.

            Args:
                x: x
            """
            return x

        def one_alphanumeric_string_arg(x: str) -> str:
            """A test function.

            Args:
                x: x
            """
            return x

        def two_arg(x: int, y: str) -> int:
            """A test function.

            Args:
                x: x
                y: y
            """
            return x + len(y)

        def one_optional_arg(x: int = 1) -> int:
            """A test function.

            Args:
                x: x with a default.
            """
            return x

        def mixed_required_args(x: int, y: int = 1) -> int:
            """A test function.

            Args:
                x: x
                y: y
            """
            return x + y

        def two_string_args(x: str, y: str) -> str:
            """A test function.

            Args:
                x: x
                y: y
            """
            return x + y

        def three_string_args(x: str, y: str, z: str) -> str:
            """A test function.

            Args:
                x: x
                y: y
                z: z
            """
            return x + y + z

        def mixed_string_int_args(x: str, y: int, z: str) -> str:
            """A test function.

            Args:
                x: x
                y: y
                z: z
            """
            return (x + z) * y

        def complex_optionals(
            x: str, y: str, a: int | None = None, b: int | None = None
        ) -> str:
            """A complex test function with optionals.

            Args:
                x: x
                y: y
                a: a
                b: b
            """
            return x + y + str(a) + str(b)

        tools = [
            Tool.from_function(zero_arg),
            Tool.from_function(one_arg),
            Tool.from_function(one_string_arg),
            Tool.from_function(one_alphanumeric_string_arg),
            Tool.from_function(two_arg),
            Tool.from_function(one_optional_arg),
            Tool.from_function(mixed_required_args),
            Tool.from_function(two_string_args),
            Tool.from_function(three_string_args),
            Tool.from_function(mixed_string_int_args),
            Tool.from_function(complex_optionals),
        ]

        with subtests.test("no args"):
            message = Message(
                content="""

        Of course, I would be happy to follow your format.

        This will be fun!

        I cannot wait to begin!!!

        I had a thought? What if I don't follow it

        OK... I will:

        Thought: you should always think about what to do
        Action: zero_arg
        Action Input:
        Observation: the result of the action

        """
            )
            tool_call = parse_message(message, tools).tool_calls[0]
            assert tool_call.function.name == "zero_arg"
            assert not tool_call.function.arguments

        with subtests.test("one arg"):
            message = Message(
                content="""

        Of course, I would be happy to follow your format.

            Thought:
            Action: one_arg
            Action Input: 1
        """
            )
            tool_call = parse_message(message, tools).tool_calls[0]
            assert tool_call.function.name == "one_arg"
            assert tool_call.function.arguments == {"x": 1}

        with subtests.test("one string arg"):
            message = Message(
                content="""

        Of course, I would be happy to follow your format.

            Thought:
            Action: one_string_arg
            Action Input: Bertrand Russell
        """
            )
            tool_call = parse_message(message, tools).tool_calls[0]
            assert tool_call.function.name == "one_string_arg"
            assert tool_call.function.arguments == {"x": "Bertrand Russell"}

        with subtests.test("one alphanumeric string arg"):
            message = Message(
                content="""

        Of course, I would be happy to follow your format.

            Thought:
            Action: one_alphanumeric_string_arg
            Action Input: R2D2 from Star Wars
        """
            )
            tool_call = parse_message(message, tools).tool_calls[0]
            assert tool_call.function.name == "one_alphanumeric_string_arg"
            assert tool_call.function.arguments == {"x": "R2D2 from Star Wars"}

        with subtests.test("two args"):
            message = Message(
                content="""
            Hey, here's some code:

            ```js
            console.log("hello world!"!)
            ```

            just thought you'd want to know!!!!

            Thought: A long thought
            what do I do next???
            Oh right
            Note: spacing below is part of test
            Action:   two_arg
            Action Input: 1, 2
            """
            )
            tool_call = parse_message(message, tools).tool_calls[0]
            assert tool_call.function.name == "two_arg"
            assert tool_call.function.arguments == {"x": 1, "y": 2}

        with subtests.test("nested args"):
            # check that parser can handle if the action input
            # is itself already a tuple
            message = Message(
                content="""
            Hey, here's some code:

            ```js
            console.log("hello world!"!)
            ```

            just thought you'd want to know!!!!

            Thought: A long thought
            what do I do next???
            Oh right
            Action:one_arg
            Action Input: (1,)
            """
            )
            tool_call = parse_message(message, tools).tool_calls[0]
            assert tool_call.function.name == "one_arg"
            assert tool_call.function.arguments == {"x": 1}

        with subtests.test("optional arg"):
            message = Message(
                content="""
            Thought: A long thought
            what do I do next???
            Oh right
            Action: one_optional_arg
            Action Input: ()
            """
            )
            tool_call = parse_message(message, tools).tool_calls[0]
            assert tool_call.function.name == "one_optional_arg"
            assert not tool_call.function.arguments

        with subtests.test("mixed args"):
            message = Message(
                content="""
            Thought: A long thought
            what do I do next???
            Oh right
            Action: mixed_required_args
            Action Input: (1,)
            """
            )
            tool_call = parse_message(message, tools).tool_calls[0]
            assert tool_call.function.name == "mixed_required_args"
            assert tool_call.function.arguments == {"x": 1}

        with subtests.test("two mixed args"):
            message = Message(
                content="""
            Thought: A long thought
            what do I do next???
            Oh right
            Action: mixed_required_args
            Action Input: (1,2)
            """
            )
            tool_call = parse_message(message, tools).tool_calls[0]
            assert tool_call.function.name == "mixed_required_args"
            assert tool_call.function.arguments == {"x": 1, "y": 2}

        with subtests.test("two string args"):
            message = Message(
                content="""
            Thought: A long thought
            what do I do next???
            Oh right
            Action: two_string_args
            Action Input: (Bertrand Russell, Friedrich Nietzsche)
            """
            )
            tool_request_message = parse_message(message, tools)
            assert isinstance(tool_request_message, ToolRequestMessage)
            assert tool_request_message.tool_calls[0].function.name == "two_string_args"
            assert tool_request_message.tool_calls[0].function.arguments == {
                "x": "Bertrand Russell",
                "y": "Friedrich Nietzsche",
            }

        with subtests.test("three string args"):
            message = Message(
                content="""
            Thought: A long thought
            what do I do next???
            Oh right
            Action: three_string_args
            Action Input: (Bertrand Russell, Friedrich Nietzsche, Immanuel Kant)
            """
            )
            tool_request_message = parse_message(message, tools)
            assert isinstance(tool_request_message, ToolRequestMessage)
            assert (
                tool_request_message.tool_calls[0].function.name == "three_string_args"
            )
            assert tool_request_message.tool_calls[0].function.arguments == {
                "x": "Bertrand Russell",
                "y": "Friedrich Nietzsche",
                "z": "Immanuel Kant",
            }

        with subtests.test("mixed string and integer args"):
            message = Message(
                content="""
            Thought: A long thought
            what do I do next???
            Oh right
            Action: mixed_string_int_args
            Action Input: (Bertrand Russell, 2, Immanuel Kant)
            """
            )
            tool_request_message = parse_message(message, tools)
            assert isinstance(tool_request_message, ToolRequestMessage)
            assert (
                tool_request_message.tool_calls[0].function.name
                == "mixed_string_int_args"
            )
            assert tool_request_message.tool_calls[0].function.arguments == {
                "x": "Bertrand Russell",
                "y": 2,
                "z": "Immanuel Kant",
            }

        with subtests.test("Attempting to put it into python ticks"):
            text = """
                ```python
                Thought: seq-2f794d237c85 contains the DsRed gene, a variant of RFP. I need to extract this gene and clone it into an E. coli-compatible vector. First, I'll slice out the RFP gene.

                Action: complex_optionals
                Action Input: ann-seq-5d7de2e1-f32a, EGE3L, 1697, 2372
                ```"""

            message = Message(content=text)
            tool_request_message = parse_message(message, tools)

        with subtests.test("A bunch of actions still only calls one tool"):
            text = """
                ```python
                Thought: seq-2f794d237c85 contains the DsRed gene, a variant of RFP. I need to extract this gene and clone it into an E. coli-compatible vector. First, I'll slice out the RFP gene.

                Action: complex_optionals
                Action Input: ann-seq-5d7de2e1-f32a, EGE3L, 1697, 2372

                Action: complex_optionals
                Action Input: ann-seq-5d7de2e1-f32a, EGE3L, 1697, 2372

                Action: complex_optionals
                Action Input: ann-seq-5d7de2e1-f32a, EGE3L, 1697, 2372
                ```"""

            message = Message(content=text)
            tool_request_message = parse_message(message, tools)
            assert len(tool_request_message.tool_calls) == 1

        with subtests.test("a single integer arg"):
            _, tools = await DummyEnv().reset()
            test_message = Message(
                role="assistant",
                content="Thought: I think.\nAction: cast_int\nAction Input: (1.1)",
            )
            parsed_message = parse_message(test_message, tools)
            assert isinstance(parsed_message, ToolRequestMessage)
            assert parsed_message.tool_calls[0].function.name == "cast_int"
            assert parsed_message.tool_calls[0].function.arguments == {"x": 1.1}
