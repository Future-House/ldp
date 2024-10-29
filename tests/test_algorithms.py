import pytest
from aviary.core import DummyEnv

from ldp.agent import SimpleAgent
from ldp.utils import discounted_returns


@pytest.mark.asyncio
async def test_rollout_and_discounting(dummy_env: DummyEnv) -> None:
    obs, tools = await dummy_env.reset()

    agent = SimpleAgent(tools=tools)
    agent_state = await agent.init_state(tools=tools)

    observations = []
    actions = []
    rewards = []
    terms = []
    done = True
    for i in range(3):  # noqa: B007
        if done:
            obs, _ = await dummy_env.reset()
            agent_state = await agent.init_state(tools=tools)

        observations.append((obs, agent_state))
        action, agent_state, _ = await agent.get_asv(agent_state, obs)
        obs, reward, done, _ = await dummy_env.step(action.value)
        actions.append(action)
        rewards.append(reward)
        terms.append(done)

    print(terms)
    d_returns = discounted_returns(rewards, terms, 0.5)
    print(d_returns)
