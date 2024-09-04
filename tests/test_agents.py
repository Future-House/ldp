import contextlib
import shutil
import tempfile
from enum import IntEnum, auto
from functools import partial
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import pytest
from aviary.env import DummyEnv
from aviary.message import Message
from aviary.tools import Tool, ToolCall, ToolRequestMessage
from httpx import AsyncClient
from pydantic import BaseModel, Field

from ldp.agent import (
    Agent,
    AgentConfig,
    DQNAgent,
    HTTPAgentClient,
    MemoryAgent,
    ReActAgent,
    SimpleAgent,
    SimpleAgentState,
    make_simple_agent_server,
)
from ldp.alg.algorithms import to_network
from ldp.graph.common_ops import LLMCallOp
from ldp.graph.gradient_estimators import llm_straight_through_estimator as llm_ste
from ldp.graph.gradient_estimators import straight_through_estimator as ste
from ldp.graph.memory import Memory
from ldp.graph.modules import ReActModule, ToolDescriptionMethods
from ldp.graph.op_utils import eval_mode
from ldp.graph.ops import OpResult
from ldp.llms import LLMModel

from . import CILLMModelNames
from .conftest import IN_GITHUB_ACTIONS

HERE = Path(__file__).parent


def intuitive_arg(x: str) -> float:  # type: ignore[empty-body]
    """Cast the input argument x to a float."""


class StubState(BaseModel):
    """Stub model docstring."""

    defaulted_int: int = Field(default=1, description="A description of the int.")
    required_str: str = Field(description="A description of the str.")


class StubEnum(IntEnum):
    """Stub enum docstring."""

    STUB1 = auto()
    STUB2 = auto()


def many_edge_cases(
    x: int,
    y: None,
    union: int | None,
    pydantic_model: StubState,
    basic_dict: dict[str, int],
    complex_dict: dict[str, tuple[str, int]],
    enum: StubEnum,
    defaulted_str: str = "default",
    defaulted_float: float = 1.0,
) -> None:
    """
    Check using docstrings as partial f-string templates like so: {summary_format}.

    Args:
        x: Yes, I end with a colon :
        y: I am null.
            And despite that there is a multiline argument description.
        union: I am a union and the current year is {current_year}.
        pydantic_model: I am a Pydantic model.
        basic_dict: I am a dictionary with primitive values.
        complex_dict: I am a dictionary with complex values.
        enum: I am an enum.
        defaulted_str: I have a string default value.
        defaulted_float: I have a float default value.
    """


class TestAgentState:
    @pytest.mark.parametrize(
        "agent",
        [SimpleAgent(), MemoryAgent(), ReActAgent(), DQNAgent()],
    )
    @pytest.mark.asyncio
    async def test_no_state_mutation(self, dummy_env: DummyEnv, agent: Agent) -> None:
        obs, tools = await dummy_env.reset()

        agent_state = agent_state_0 = await agent.init_state(tools=tools)
        agent_state_0_json = agent_state_0.model_dump_json()
        for _ in range(3):  # Give a few steps to finish, the assertion needs >0 steps
            action, agent_state, _ = await agent.get_asv(agent_state, obs)
            obs, reward, done, truncated = await dummy_env.step(action.value)
            if done:
                break

        assert (
            agent_state_0_json == agent_state_0.model_dump_json()
        ), "Agent state should not be mutated between calls to get_asv"

    def test_serialization_deserializaton(self) -> None:
        orig_state = SimpleAgentState(
            messages=[Message(content="stub"), ToolRequestMessage(content="stub2")]
        )
        copied_state = SimpleAgentState(**orig_state.model_dump())
        assert orig_state.messages == copied_state.messages
        assert isinstance(copied_state.messages[1], ToolRequestMessage)


class TestSimpleAgent:
    @pytest.mark.parametrize(
        "model_name", [CILLMModelNames.ANTHROPIC.value, CILLMModelNames.OPENAI.value]
    )
    @pytest.mark.asyncio
    @pytest.mark.flaky(  # Rerun if LLM call does not return expected result
        reruns=3, only_on=[AssertionError]
    )
    async def test_dummyenv(self, dummy_env: DummyEnv, model_name: str) -> None:
        obs, tools = await dummy_env.reset()

        agent = SimpleAgent(llm_model={"model": model_name, "temperature": 0.1})
        agent_state = await agent.init_state(tools=tools)
        action, agent_state, _ = await agent.get_asv(agent_state, obs)
        obs, reward, done, truncated = await dummy_env.step(action.value)
        assert (
            reward > 0
        ), "Reward should be positive, indicating agent called print_story tool"
        assert done

        # Check serialization after get_asv runs to ensure private
        # Ops aren't included
        assert agent.model_dump() == {
            "llm_model": {"model": model_name, "temperature": 0.1},
            "sys_prompt": None,
        }

        # Check we can get the LLM results to sum cost and count tokens
        assert action.call_id is not None, "Compute graph not attached to action."
        for op_r in action.traverse():
            if issubclass(op_r.op_class, LLMCallOp):
                # will raise if cannot retrieve result
                op_r._get_from_ctx("result")
                break
        else:
            raise RuntimeError("Could not find LLMCallOp in compute graph")

    @pytest.mark.parametrize(
        "model_name", [CILLMModelNames.ANTHROPIC.value, CILLMModelNames.OPENAI.value]
    )
    @pytest.mark.asyncio
    @pytest.mark.flaky(  # Rerun if LLM call does not return expected result
        reruns=3, only_on=[AssertionError]
    )
    async def test_agent_grad(self, dummy_env: DummyEnv, model_name: str) -> None:
        obs, tools = await dummy_env.reset()

        agent = SimpleAgent(llm_model={"model": model_name, "temperature": 0.1})
        agent_state = await agent.init_state(tools=tools)
        action, agent_state, _ = await agent.get_asv(agent_state, obs)
        assert action.call_id is not None
        obs, reward, done, _ = await dummy_env.step(action.value)
        assert (
            reward > 0
        ), "Reward should be positive, indicating agent called print_story tool"
        assert done
        assert (
            action.call_id is not None
        ), "action is not associated with a forward pass call_id"

        # NOTE: we would not normally pass reward as a gradient, but this is a way
        # to check that gradients are flowing
        action.compute_grads(
            reward,
            backward_fns={"_config_op": ste, "_llm_call_op": llm_ste},
        )
        _, g = action.ctx.get_input_grads(action.call_id)
        assert isinstance(
            g["config"], dict
        ), "compute_grads() didn't descend into config dict"
        assert all(g["config"].values()), "Gradient should be non-zero"

        graph = to_network(action)
        with (
            tempfile.NamedTemporaryFile(mode="w", suffix=".png") as f,
            contextlib.suppress(
                FileNotFoundError  # Allow tests to run without graphviz on OS
            ),
        ):
            nx.drawing.nx_pydot.to_pydot(graph).write_png(f.name)

            if not IN_GITHUB_ACTIONS:
                output_dir = HERE / "test_outputs"
                output_dir.mkdir(exist_ok=True)
                shutil.copy(
                    f.name,
                    output_dir / f"TestSimpleAgent.test_agent_grad.{model_name}.png",
                )


class TestMemoryAgent:
    # # On 5/14/2024, claude 3 opus would not follow its past memories
    @pytest.mark.parametrize("model_name", [CILLMModelNames.OPENAI.value])
    @pytest.mark.asyncio
    @pytest.mark.flaky(  # Rerun if LLM call does not return expected result
        reruns=3, only_on=[AssertionError]
    )
    async def test_dummyenv(self, dummy_env: DummyEnv, model_name: str) -> None:
        obs, tools = await dummy_env.reset()

        agent = MemoryAgent(llm_model={"model": model_name, "temperature": 0.1})
        agent_state = await agent.init_state(tools=tools)

        # access memory and add one to it
        action = ToolRequestMessage(
            content="Stories that start with 'Once there was' are always interesting.",
            tool_calls=[
                ToolCall.from_name(
                    tools[0].info.name, story="Once there were was nothing."
                )
            ],
        )
        memory = agent._memory_op.memory_model
        await memory.add_memory(
            Memory(
                query="Write a 5 word story and call print",
                output=str(action),
                value=1000.0,
            )
        )

        new_action, agent_state, _ = await agent.get_asv(agent_state, obs)
        assert "Once there was" in str(new_action)
        obs, reward, done, truncated = await dummy_env.step(new_action.value)
        assert (
            reward > 0
        ), "Reward should be positive, indicating agent called print_story tool"
        assert done

        # Check we can get the LLM results to sum cost and count tokens
        assert new_action.call_id is not None, "Compute graph not attached to action."
        for op_r in new_action.traverse():
            if issubclass(op_r.op_class, LLMCallOp):
                # will raise if cannot retrieve result
                op_r._get_from_ctx("result")
                break
        else:
            raise RuntimeError("Could not find LLMCallOp in compute graph")

    @pytest.mark.asyncio
    @pytest.mark.flaky(  # Rerun if LLM call does not return expected result
        reruns=3, only_on=[AssertionError]
    )
    async def test_agent_grad(self, dummy_env: DummyEnv) -> None:
        obs, tools = await dummy_env.reset()

        agent = MemoryAgent()
        agent_state = await agent.init_state(tools=tools)
        action, agent_state, _ = await agent.get_asv(agent_state, obs)
        assert action.call_id is not None
        obs, reward, done, truncated = await dummy_env.step(action.value)
        assert (
            reward > 0
        ), "Reward should be positive, indicating agent called print_story tool"
        assert done
        assert (
            action.call_id is not None
        ), "action is not associated with a forward pass call_id"

        # NOTE: we would not normally pass reward as a gradient, but this is a way
        # to check that gradients are flowing
        ste_ = partial(ste, descend=False)
        action.compute_grads(
            reward,
            backward_fns={
                "_prompt_op": ste_,
                "_package_op": ste_,
                "_format_memory_op": ste_,
                "_memory_op": ste_,
                "_llm_call_op": llm_ste,
            },
        )
        _, g = action.ctx.get_input_grads(action.call_id)
        assert isinstance(
            g["config"], dict
        ), "compute_grads() didn't descend into config dict"
        assert all(g["config"].values()), "Action gradient should be non-zero"

        memory_op = agent._memory_op
        mem_call_ids = list(memory_op.get_call_ids({action.call_id.run_id}))
        assert len(mem_call_ids) == 1, "MemoryOp should have been called exactly once"
        _, g = memory_op.get_input_grads(mem_call_ids[0])
        assert any(g.values()), "Memory gradient should be non-zero"


class TestReActAgent:
    @pytest.mark.parametrize(
        "model_name", [CILLMModelNames.ANTHROPIC.value, "gpt-4-turbo"]
    )
    @pytest.mark.asyncio
    @pytest.mark.flaky(  # Rerun if LLM call does not return expected result
        reruns=3, only_on=[AssertionError]
    )
    async def test_react_dummyenv(self, dummy_env: DummyEnv, model_name: str) -> None:
        obs, tools = await dummy_env.reset()
        agent = ReActAgent(llm_model={"model": model_name, "temperature": 0.1})
        agent_state = await agent.init_state(tools=tools)
        action, agent_state, _ = await agent.get_asv(agent_state, obs)
        obs, reward, done, truncated = await dummy_env.step(action.value)
        assert (
            reward > 0
        ), "Reward should be positive, indicating agent called print_story tool"
        assert done

        # Check we can get the LLM results to sum cost and count tokens
        assert action.call_id is not None, "Compute graph not attached to action."
        for op_r in action.traverse():
            if issubclass(op_r.op_class, LLMCallOp):
                # will raise if cannot retrieve result
                op_r._get_from_ctx("result")
                break
        else:
            raise RuntimeError("Could not find LLMCallOp in compute graph")

    @pytest.mark.asyncio
    async def test_multi_step(self, dummy_env: DummyEnv) -> None:
        obs, tools = await dummy_env.reset()
        obs = dummy_env.state.messages = [
            Message(
                content=(
                    "Cast '5.5' to a float, then to an integer,"
                    " and finally use it to write a story of that many words."
                )
            )
        ]
        agent = ReActAgent()
        agent_state = await agent.init_state(tools=tools)
        for i in range(4):  # noqa: B007
            action, agent_state, _ = await agent.get_asv(agent_state, obs)
            for m in agent_state.messages:
                assert m.content
                assert (
                    "Observation: Observation" not in m.content
                ), "Prepended duplicate observations"
            obs, _, done, _ = await dummy_env.step(action.value)
            if done:
                break
        if i < 2 or not done:
            raise AssertionError(
                "Environment should have finished, with at least 2 environment steps."
            )

    def test_agent_op_naming(self) -> None:
        agent = ReActAgent()
        for op_name in (
            "prompt_op",
            "package_msg_op",
            "tool_select_module.config_op",
            "tool_select_module.llm_call_op",
            "tool_select_module.parse_msg_op",
        ):
            obj, expected = agent._react_module, f"_react_module.{op_name}"
            if "." in op_name:
                op, op_name = op_name.split(".", maxsplit=1)
                obj = getattr(obj, op)
            assert getattr(obj, op_name).name == expected

    @pytest.mark.parametrize(
        "model_name", [CILLMModelNames.ANTHROPIC.value, "gpt-4-turbo"]
    )
    @pytest.mark.asyncio
    @pytest.mark.flaky(  # Rerun if LLM call does not return expected result
        reruns=3, only_on=[AssertionError]
    )
    async def test_agent_grad(self, dummy_env: DummyEnv, model_name: str) -> None:
        obs, tools = await dummy_env.reset()

        agent = ReActAgent(llm_model={"model": model_name, "temperature": 0.1})
        agent_state = await agent.init_state(tools=tools)
        action, agent_state, _ = await agent.get_asv(agent_state, obs)
        assert action.call_id is not None
        obs, reward, done, truncated = await dummy_env.step(action.value)
        assert (
            reward > 0
        ), "Reward should be positive, indicating agent called print_story tool"
        assert done
        assert (
            action.call_id is not None
        ), "action is not associated with a forward pass call_id"

        # NOTE: we would not normally pass reward as a gradient, but this is a way
        # to check that gradients are flowing
        ste_ = partial(ste, descend=False)
        action.compute_grads(
            reward,
            # Give everything a straight-through gradient approximation
            # so we can confirm gradient flow
            backward_fns={
                "_react_module.package_msg_op": ste_,
                "_react_module.prompt_op": ste_,
                "_react_module.tool_select_module.llm_call_op": llm_ste,
                "_react_module.tool_select_module.parse_msg_op": ste_,
            },
        )
        _, g = action.ctx.get_input_grads(action.call_id)
        assert all(g.values()), "Gradient should be non-zero"

        # make sure it propagated far enough
        prompt_op = agent._react_module.prompt_op
        _, g = prompt_op.get_input_grads(
            next(iter(prompt_op.get_call_ids({action.call_id.run_id})))
        )
        assert all(g.values()), "PromptOp gradients should be positive"

        graph = to_network(action, max_label_height=4, max_label_width=50)
        with (
            tempfile.NamedTemporaryFile(mode="w", suffix=".png") as f,
            contextlib.suppress(
                FileNotFoundError  # Allow tests to run without graphviz on OS
            ),
        ):
            nx.drawing.nx_pydot.to_pydot(graph).write_png(f.name)

            if not IN_GITHUB_ACTIONS:
                output_dir = HERE / "test_outputs"
                output_dir.mkdir(exist_ok=True)
                shutil.copy(
                    f.name,
                    output_dir / f"TestReActAgent.test_agent_grad.{model_name}.png",
                )

    @pytest.mark.parametrize(
        ("description_method", "expected"),
        [
            (ToolDescriptionMethods.STR, NotImplementedError),
            (
                ToolDescriptionMethods.JSON,
                (
                    "Answer the following questions as best you can. You have access to"
                    " the following tools:\n\nTools are specified with a JSON"
                    ' schema.\n{"name":"many_edge_cases","description":"Check using'
                    " docstrings as partial f-string templates like so:"
                    ' {summary_format}.","parameters":{"type":"object","properties":{"x":{"description":"Yes,'
                    " I end with a colon"
                    ' :","title":"X","type":"integer"},"y":{"description":"I am null.'
                    " And despite that there is a multiline argument"
                    ' description.","title":"Y","type":"null"},"union":{"anyOf":[{"type":"integer"},{"type":"null"}],"description":"I'
                    " am a union and the current year is"
                    ' {current_year}.","title":"Union"},"pydantic_model":{"allOf":[{"$ref":"#/$defs/StubState"}],"description":"I'
                    " am a Pydantic"
                    ' model."},"basic_dict":{"additionalProperties":{"type":"integer"},"description":"I'
                    ' am a dictionary with primitive values.","title":"Basic'
                    ' Dict","type":"object"},"complex_dict":{"additionalProperties":{"maxItems":2,"minItems":2,"prefixItems":[{"type":"string"},{"type":"integer"}],"type":"array"},"description":"I'
                    ' am a dictionary with complex values.","title":"Complex'
                    ' Dict","type":"object"},"enum":{"allOf":[{"$ref":"#/$defs/StubEnum"}],"description":"I'
                    " am an"
                    ' enum."},"defaulted_str":{"default":"default","description":"I'
                    ' have a string default value.","title":"Defaulted'
                    ' Str","type":"string"},"defaulted_float":{"default":1.0,"description":"I'
                    ' have a float default value.","title":"Defaulted'
                    ' Float","type":"number"}},"required":["x","y","union","pydantic_model","basic_dict","complex_dict","enum"],"$defs":{"StubEnum":{"description":"Stub'
                    " enum"
                    ' docstring.","enum":[1,2],"title":"StubEnum","type":"integer"},"StubState":{"description":"Stub'
                    " model"
                    ' docstring.","properties":{"defaulted_int":{"default":1,"description":"A'
                    ' description of the int.","title":"Defaulted'
                    ' Int","type":"integer"},"required_str":{"description":"A'
                    ' description of the str.","title":"Required'
                    ' Str","type":"string"}},"required":["required_str"],"title":"StubState","type":"object"}}}}\n{"name":"intuitive_arg","description":"Cast'
                    " the input argument x to a"
                    ' float.","parameters":{"type":"object","properties":{"x":{"title":"X","type":"string"}},"required":["x"]}}\n\nUse'
                    " the following format:\n\nThought: you should always think about"
                    " what to do\nAction: the action to take, should be one of"
                    " [many_edge_cases, intuitive_arg]\nAction Input: comma separated"
                    " list of inputs to action as python tuple\nObservation: the result"
                    " of the action\n... (this Thought/Action/Action Input/Observation"
                    " can repeat N times)\n\nExample:\n\nThought: I need to use the"
                    ' get_weather tool\nAction: get_weather\nAction Input: "New York",'
                    " 7\nObservation: The 7 day forecast for New York is [...]"
                ),
            ),
            (
                ToolDescriptionMethods.XML,
                (
                    "Answer the following questions as best you can. You have access to"
                    " the following tools:\n\nTools are specified with an XML"
                    " schema.\n<function_info><name>many_edge_cases</name><description>Check"
                    " using docstrings as partial f-string templates like so:"
                    " {summary_format}.</description><parameters><type>object</type><properties><x><description>Yes,"
                    " I end with a colon"
                    " :</description><title>X</title><type>integer</type></x><y><description>I"
                    " am null. And despite that there is a multiline argument"
                    " description.</description><title>Y</title><type>null</type></y><union><anyOf><item><type>integer</type></item><item><type>null</type></item></anyOf><description>I"
                    " am a union and the current year is"
                    " {current_year}.</description><title>Union</title></union><pydantic_model><allOf><item><key"
                    ' name="$ref">#/$defs/StubState</key></item></allOf><description>I'
                    " am a Pydantic"
                    " model.</description></pydantic_model><basic_dict><additionalProperties><type>integer</type></additionalProperties><description>I"
                    " am a dictionary with primitive values.</description><title>Basic"
                    " Dict</title><type>object</type></basic_dict><complex_dict><additionalProperties><maxItems>2</maxItems><minItems>2</minItems><prefixItems><item><type>string</type></item><item><type>integer</type></item></prefixItems><type>array</type></additionalProperties><description>I"
                    " am a dictionary with complex values.</description><title>Complex"
                    " Dict</title><type>object</type></complex_dict><enum><allOf><item><key"
                    ' name="$ref">#/$defs/StubEnum</key></item></allOf><description>I'
                    " am an"
                    " enum.</description></enum><defaulted_str><default>default</default><description>I"
                    " have a string default value.</description><title>Defaulted"
                    " Str</title><type>string</type></defaulted_str><defaulted_float><default>1.0</default><description>I"
                    " have a float default value.</description><title>Defaulted"
                    " Float</title><type>number</type></defaulted_float></properties><required><item>x</item><item>y</item><item>union</item><item>pydantic_model</item><item>basic_dict</item><item>complex_dict</item><item>enum</item></required><key"
                    ' name="$defs"><StubEnum><description>Stub enum'
                    " docstring.</description><enum><item>1</item><item>2</item></enum><title>StubEnum</title><type>integer</type></StubEnum><StubState><description>Stub"
                    " model"
                    " docstring.</description><properties><defaulted_int><default>1</default><description>A"
                    " description of the int.</description><title>Defaulted"
                    " Int</title><type>integer</type></defaulted_int><required_str><description>A"
                    " description of the str.</description><title>Required"
                    " Str</title><type>string</type></required_str></properties><required><item>required_str</item></required><title>StubState</title><type>object</type></StubState></key></parameters></function_info>\n<function_info><name>intuitive_arg</name><description>Cast"
                    " the input argument x to a"
                    " float.</description><parameters><type>object</type><properties><x><title>X</title><type>string</type></x></properties><required><item>x</item></required></parameters></function_info>\n\nUse"
                    " the following format:\n\nThought: you should always think about"
                    " what to do\nAction: the action to take, should be one of"
                    " [many_edge_cases, intuitive_arg]\nAction Input: comma separated"
                    " list of inputs to action as python tuple\nObservation: the result"
                    " of the action\n... (this Thought/Action/Action Input/Observation"
                    " can repeat N times)\n\nExample:\n\nThought: I need to use the"
                    ' get_weather tool\nAction: get_weather\nAction Input: "New York",'
                    " 7\nObservation: The 7 day forecast for New York is [...]"
                ),
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_complex_system_prompt(
        self,
        description_method: ToolDescriptionMethods,
        expected: str | type[Exception],
    ) -> None:
        tools = [
            Tool.from_function(many_edge_cases),
            Tool.from_function(intuitive_arg, allow_empty_param_descriptions=True),
        ]
        user_msg = Message(content="Cast the string '5.6' to a float.")
        with (
            patch.object(LLMModel, "achat") as mock_achat,
            patch.object(ReActModule, "parse_message"),
        ):
            agent = ReActAgent(tool_description_method=description_method)
            agent_state = await agent.init_state(tools=tools)
            if not isinstance(expected, str):
                with pytest.raises(expected):
                    await agent.get_asv(agent_state, obs=[user_msg])
                return
            await agent.get_asv(agent_state, obs=[user_msg])
        mock_achat.assert_awaited_once()
        assert mock_achat.await_args
        assert mock_achat.await_args[0][0] == [
            Message(role="system", content=expected),
            user_msg,
        ]


class TestHTTPAgentClient:
    @patch.dict("os.environ", {"AUTH_TOKEN": "stub"})
    @pytest.mark.asyncio
    @pytest.mark.flaky(  # Rerun if LLM call does not return expected result
        reruns=3, only_on=[AssertionError]
    )
    async def test_lifecycle(self, dummy_env: DummyEnv) -> None:
        obs, tools = await dummy_env.reset()
        # Let's turn the prompt to require multiple steps
        obs = dummy_env.state.messages = [
            Message(
                content=(
                    "Cast '5.5' to a float, then to an integer,"
                    " and finally use it to write a story of that many words."
                )
            )
        ]

        remote_agent = SimpleAgent()
        base_server_url = "http://testserver"
        agent_client = HTTPAgentClient[SimpleAgentState](
            agent_state_type=SimpleAgentState,
            server_url=base_server_url,
            request_headers={"Authorization": "Bearer stub"},
        )
        async with AsyncClient(
            app=make_simple_agent_server(agent=remote_agent),
            base_url=base_server_url,
        ) as async_client:
            # NOTE: just directly hit the server, since info is not an Agent method
            response = await async_client.get(
                f"{base_server_url}/info", headers={"Authorization": "Bearer stub"}
            )
            response.raise_for_status()
            assert response.json()["agent_type"] == "SimpleAgent"

            with patch("httpx.AsyncClient.post", async_client.post):
                agent_state_0 = await agent_client.init_state(tools=tools)
            assert [t.info for t in agent_state_0.tools] == [t.info for t in tools]

            # NOTE: also check we can repeatedly call the get_asv
            with patch("httpx.AsyncClient.post", async_client.post), eval_mode():
                await agent_client.get_asv(agent_state_0, obs)
                await agent_client.get_asv(agent_state_0, obs)
            with patch("httpx.AsyncClient.post", async_client.post), eval_mode():
                action, agent_state_1, vhat = await agent_client.get_asv(
                    agent_state_0, obs
                )

            assert isinstance(action, OpResult)
            assert isinstance(action.value, ToolRequestMessage)
            assert isinstance(agent_state_1, SimpleAgentState)
            assert len(agent_state_1.messages) == 2
            assert isinstance(agent_state_1.messages[0], Message)
            assert isinstance(agent_state_1.messages[1], ToolRequestMessage)
            assert isinstance(vhat, float)

            # This makes an obs with ToolResponseMessage inside
            obs, reward, done, _ = await dummy_env.step(action.value)
            assert not done

            with patch("httpx.AsyncClient.post", async_client.post), eval_mode():
                # Check we can make a second sequential Agent decision without crashing
                await agent_client.get_asv(agent_state_1, obs)


class TestDQNAgent:
    @pytest.mark.parametrize(
        "model_name", ["gpt-4-turbo", CILLMModelNames.ANTHROPIC.value]
    )
    @pytest.mark.asyncio
    @pytest.mark.flaky(  # Rerun if LLM call does not return expected result
        reruns=3, only_on=[AssertionError]
    )
    async def test_dummyenv(self, dummy_env: DummyEnv, model_name: str) -> None:
        obs, tools = await dummy_env.reset()
        agent = DQNAgent(llm_strategy_model={"model": model_name, "temperature": 1.0})
        agent_state = await agent.init_state(tools)
        action, agent_state, _ = await agent.get_asv(agent_state, obs)
        obs, reward, done, truncated = await dummy_env.step(action.value)
        assert (
            reward > 0
        ), "Reward should be positive, indicating agent called print_story tool"
        assert done


@pytest.mark.parametrize("agent_cls", [SimpleAgent, DQNAgent, MemoryAgent, ReActAgent])
def test_agent_config(agent_cls: type[Agent]):
    config = AgentConfig(agent_type=agent_cls.__name__)
    assert isinstance(hash(config), int), "AgentConfig should be hashable"
    agent = config.construct_agent()
    assert isinstance(agent, agent_cls)