# ldp

An framework for constructing language model agents and training on constructive tasks.

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

## Agent

An agent is something that interacts with an environment (defined in our other GitHub repo [Future-House/aviary](https://github.com/Future-House/aviary)).

An agent uses tools in response to observations, which are just natural language observations. An agent has two functions:

```py
agent_state = await agent.init_state(tools=tools)
new_action, new_agent_state, value = await agent.get_asv(agent_state, obs)
```

`get_asv(agent_state, obs)` chooses an action (`a`) conditioned on the observation messages,
and returns the next agent state (`s`) and a value estimate (`v`).
The first argument, `agent_state`, is a state specific for the agent.
The state is outside of the agent so agents are functional, enabling batching across environments.
You can make the state `None` if you aren't using it. It could contain things like memory, as a list of previous observations and actions.

The `obs` are not the complete list of all prior observations, but rather the return of `env.step`.
Usually the state should keep track of these.

Value is the agent's state-action value estimate; it can default to 0.
This is used for training with reinforcement learning.

## Computing Actions

You can just emit actions directly if you want:

```py
from aviary.core import ToolCall


def get_asv(agent_state, obs):
    action = ToolCall.from_name("calculator_tool", x="3 * 2")
    return action, agent_state, 0
```

but likely you want to do something more sophisticated.
Here's how our `SimpleAgent` - which just relies on a single LLM call - works (typing omitted):

```py
from ldp.agent import Agent
from ldp.graph import LLMCallOp


class AgentState:
    def __init__(self, messages, tools):
        self.messages = messages
        self.tools = tools


class SimpleAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.llm_call_op = LLMCallOp()

    async def init_state(self, tools):
        return AgentState([], tools)

    async def get_asv(self, agent_state, obs):
        action = await self.llm_call_op(
            config={"name": "gpt-4o", "temperature": 0.1},
            msgs=agent_state.messages + obs,
            tools=agent_state.tools,
        )
        new_state = AgentState(
            messages=agent_state.messages + obs + [action], tools=agent_state.tools
        )
        return action, new_state, 0.0
```

Notice how it's pretty simple. We have to do some bookkeeping - namely appending messages as they come and passing tools. There is no magic here.

### Compute Graph

We do have a compute graph - which helps if you want to differentiate with respect to parameters inside your agent (including possibly the LLM). If your compute graph looks like the above example - where all you do is call an LLM directly, then don't worry about this.

If you want to do more complex agents and train them, then read on. Let's start with an example compute graph

```py
from ldp.graph import FxnOp, LLMCallOp, PromptOp, compute_graph

op_a = FxnOp(lambda x: 2 * x)

async with compute_graph():
    op_result = op_a(3)
```

This creates a compute graph and executes it. The compute graph is silly - just doubles the input. The compute graph executions and gradients are saved in a context for later use, like training updates. For example:

```py
print(op_result.compute_grads())
```

Now, inside the `SimpleAgent` example above, you can see some of the compute graph. Let's see a more complex example for an agent that has a memory it can draw upon.

```py
@compute_graph()
async def get_asv(self, agent_state, obs):
    # Update state with new observations
    next_state = agent_state.get_next_state(obs)

    # Retrieve relevant memories
    query = await self._query_factory_op(next_state.messages)
    memories = await self._memory_op(query, matches=self.num_memories)

    # Format memories and package messages
    formatted_memories = await self._format_memory_op(self.memory_prompt, memories)
    memory_prompt = await self._prompt_op(memories=formatted_memories)
    packaged_messages = await self._package_op(
        next_state.messages, memory_prompt=memory_prompt, use_memories=bool(memories)
    )

    # Make LLM call and update state
    config = await self._config_op()
    result = await self._llm_call_op(
        config, msgs=packaged_messages, tools=next_state.tools
    )
    next_state.messages.extend([result])

    return result, next_state, 0.0
```

You can see in this example that we use differentiable ops to ensure there is a connection in the compute graph from the LLM result (action) back to things like the memory retrieval and the query used to retrieve the memory.

Why use a compute graph? Aside from a gradient, using the compute graph enables the tracking of all inputs/outputs to the ops and serialization/deserialization of the compute graph so that you can easily save/load them. The tracking of input/outputs also makes it easier to do things like fine-tuning or reinforcement learning on the underlying LLMs.

## Generic Support

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

## Tutorial

See a tutorial of building and [running an agent for GSM8K](docs/agent_tutorial.ipynb)
