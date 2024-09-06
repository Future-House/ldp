# ldp

Agent framework for constructing language model agents and training on constructive tasks.

This repo models agent-environment interactions using a
[Partially Observable Markov Decision Process][pomdp] (POMDP).
Inspired by POMDP, this repo's name `ldp` stands for Language Decision Processes.

[pomdp]: https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process

## Installation

To install `ldp`:

```bash
pip install -e .
```

If you plan to export Graphviz visualizations,
make sure you also install the `graphviz` library into your OS via:

- Linux: `apt install graphviz`
- macOS: `brew install graphviz`

## Agent/Policy

An agent should have two functions:

```py
agent_state = await agent.init_state(tools=tools)
new_action, new_agent_state, value = await agent.get_asv(
    agent_state, obs
)
```

An agent should have a function `get_asv(agent_state, obs)`
that chooses an action (`a`) from the observation messages,
and returns the next agent state (`s`) and a value estimate (`v`).
The first argument, `agent_state`, is a state specific for the agent
that can be used for training from episodes.
You can make it `None` if you aren't using it.
It could contain things like agent memory.

The `obs` are not the complete list of observations, but rather the last list from `env.step`.
The agent should keep track of observations via its state if it would like to keep them.

The value can be `0`,
it is the agent's estimate of the future rewards given its state and observations.
This is used for training.

### Generic Support

The `Agent` (as well as classes in `agent.ops`)
are [generics](https://en.wikipedia.org/wiki/Generic_programming),
which means:

- `Agent` is designed to support arbitrary types
- Subclasses can exactly specify state types, making the code more readable

If you are new to Python generics (`typing.Generic`),
please read about them in [Python typing](https://docs.python.org/3/library/typing.html#generics).

Below is how to specify an agent with a custom state type.

```py
from dataclasses import dataclass, field
from datetime import datetime

from ldp.agents import Agent


@dataclass
class MyComplexState:
    vector: list[float]
    timestamp: datetime = field(default_factory=datetime.now)


class MyAgent(Agent[MyComplexState]):
    """Some agent who is now type checked to match the custom state."""
```

## Complete Example

```py
from ldp.agent import SimpleAgent
from aviary.env import DummyEnv

env = DummyEnv()
agent = SimpleAgent()

obs, tools = await env.reset()
agent_state = await agent.init_state(tools=tools)

done = False
while not done:
    action, agent_state, _ = await agent.get_asv(agent_state, obs)
    obs, reward, done, truncated = await env.step(action.value)
```
