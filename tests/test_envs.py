from importlib.metadata import version
from typing import ClassVar

import litellm
import pytest
from aviary.core import (
    DummyEnv,
    DummyEnvState,
    Message,
    Tool,
    ToolCall,
    ToolRequestMessage,
    ToolResponseMessage,
)
from lmi import CommonLLMNames

from ldp.agent import SimpleAgent


class ParallelizedDummyEnv(DummyEnv):
    def __init__(self, right_hand_broken: bool = False):
        super().__init__()
        self.right_hand_broken = right_hand_broken

    RIGHT_HAND_BROKEN_MESSAGE: ClassVar[str] = "Right hand is broken."

    async def reset(self) -> tuple[list[Message], list[Tool]]:
        def move_right_hand(
            distance: int,  # noqa: ARG001
            state: DummyEnvState,
        ) -> None:
            """
            Move your right hand forward or backward.

            Args:
                distance: Integer distance to move (mm), where forward is positive.
                state: Current state.
            """
            if self.right_hand_broken:  # Use this to test tool errors
                raise RuntimeError(self.RIGHT_HAND_BROKEN_MESSAGE)
            state.reward += 1

        def move_left_hand(
            distance: int,  # noqa: ARG001
            state: DummyEnvState,
        ) -> None:
            """
            Move your left hand forward or backward.

            Args:
                distance: Integer distance to move (mm), where forward is positive.
                state: Current state.
            """
            state.reward += 1

        def smile_and_wave(state: DummyEnvState) -> None:
            """
            Smile and wave.

            Args:
                state: Current state.
            """
            state.reward = 10
            state.done = True

        self.tools = [
            Tool.from_function(move_left_hand),
            Tool.from_function(move_right_hand),
            Tool.from_function(smile_and_wave),
        ]
        self.state = type(self).State(
            messages=[
                Message(
                    role="user",
                    content=(
                        "You are the president of the United States of America."
                        " Please move both hands at the same time, and then smile"
                        " and wave."
                    ),
                )
            ]
        )
        return self.state.messages, self.tools


class TestParallelism:
    @pytest.mark.asyncio
    @pytest.mark.vcr
    async def test_SimpleAgent_can_parallel_call(self) -> None:
        env = ParallelizedDummyEnv()
        obs, tools = await env.reset()
        agent = SimpleAgent()

        # Check parallel tool calls
        action, agent_state, _ = await agent.get_asv(  # noqa: RUF059
            await agent.init_state(tools=tools), obs
        )
        selected_tools: set[str] = {tc.function.name for tc in action.value.tool_calls}
        assert {
            "move_left_hand",
            "move_right_hand",
        } <= selected_tools, (
            f"Agent should've chosen tools in parallel, but it chose {selected_tools}"
        )

    @pytest.mark.parametrize(
        "model_name", [CommonLLMNames.ANTHROPIC_TEST.value, "gpt-4-turbo"]
    )
    @pytest.mark.asyncio
    async def test_exec_tool_calls_handling(self, model_name: str) -> None:
        env = ParallelizedDummyEnv(right_hand_broken=True)
        obs, tools = await env.reset()
        right_hand_tool = tools[1]
        agent = SimpleAgent(
            llm_model=SimpleAgent.model_fields["llm_model"].default
            | {"name": model_name}
        )
        agent_state = await agent.init_state(tools=tools)

        # 1. Let's DIY create a ToolRequestMessage for test determinism
        request_msg = ToolRequestMessage(
            content="stub", tool_calls=[ToolCall.from_tool(right_hand_tool, distance=5)]
        )
        agent_state.messages.extend([*obs, request_msg])

        # 2. Okay, our hand was broken, let's handle it DIY-style
        try:
            obs, *_ = await env.step(action=request_msg)
        except RuntimeError as exc:
            obs = [
                Message(
                    content=f"Failed to execute tools with message:\n{exc}", role="tool"
                )
            ]
        else:
            raise AssertionError("Should have blown up per the test logic.")

        # 2. Well, it looks like both Anthropic and OpenAI don't like DIY-style
        #    (using a bare Message) because they expect a tool call ID and tool name
        with pytest.raises(
            litellm.BadRequestError,
            match=(
                "invalid" if version(litellm.__name__) < "1.45.0" else "tool_call_id"
            ),
        ):
            await agent.get_asv(agent_state, obs)

        # 3. Alright, let's check the agent doesn't blow up if we use a
        #    ToolResponseMessage as Anthropic and OpenAI expect
        await agent.get_asv(
            agent_state,
            ToolResponseMessage.from_request(request_msg, contents=["null"]),  # type: ignore[arg-type]
        )

        # 4. Now that we have confirmed that, let's make sure exec_tool_calls
        #    can automate this for us
        obs = await env.exec_tool_calls(  # type: ignore[assignment]
            message=request_msg, state=env.state, handle_tool_exc=True
        )
        (failure_tool_response,) = obs
        assert isinstance(failure_tool_response, ToolResponseMessage)
        assert env.RIGHT_HAND_BROKEN_MESSAGE in failure_tool_response.content
        await agent.get_asv(agent_state, obs)
