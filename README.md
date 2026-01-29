# Language Decision Processes (LDP)

<!-- pyml disable-num-lines 10 line-length -->

[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Future-House/ldp)
[![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)
[![Docs](https://assets.readthedocs.org/static/projects/badges/passing-flat.svg)](https://futurehouse.gitbook.io/futurehouse-cookbook/ldp-language-decision-processes)
[![PyPI version](https://badge.fury.io/py/ldp.svg)](https://badge.fury.io/py/ldp)
[![tests](https://github.com/Future-House/ldp/actions/workflows/tests.yml/badge.svg)](https://github.com/Future-House/ldp)
[![CodeFactor](https://www.codefactor.io/repository/github/future-house/ldp/badge)](https://www.codefactor.io/repository/github/future-house/ldp)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue?style=flat&logo=python&logoColor=white)](https://www.python.org)

<p align="center">
  <a href="https://arxiv.org/abs/2412.21154">
    <img src="docs/assets/ldp_chessboard.png" width="300" alt="row playing chess" />
  </a>
</p>

**LDP** [^1] is a framework for enabling modular interchange of language agents, environments, and optimizers.
A language decision process (LDP)
is a partially-observable Markov decision process (POMDP)
where actions and observations consist of natural language.
The full definition from the Aviary paper [^1] is:

<p align="left">
  <a href="https://arxiv.org/abs/2412.21154">
    <img src="docs/assets/ldp_definition.png" width="600" alt="LDP definition from paper" />
  </a>
</p>

See the following [tutorial](https://github.com/Future-House/ldp/blob/main/tutorials/creating_a_language_agent.ipynb)
for an example of how to run an LDP agent.

[Overview](#overview)
| [Getting Started](#getting-started)
| [Documentation](https://futurehouse.gitbook.io/futurehouse-cookbook/ldp-language-decision-processes)
| [Paper](https://arxiv.org/abs/2412.21154)

## What's New?

- Check out our new [Tutorial](https://github.com/Future-House/ldp/blob/main/tutorials/creating_a_language_agent.ipynb)
  notebook on running an LDP agent in an Aviary environment!
- The Aviary paper has been posted to [arXiv](https://arxiv.org/abs/2412.21154)!
  Further updates forthcoming!

## Overview

<p align="left">
  <a href="https://arxiv.org/abs/2412.21154">
    <img src="docs/assets/Aviary.png" width="800" alt="Aviary and LDP overview from paper" />
  </a>
</p>

A pictorial overview of the language decision process (LDP) framework
together with five implemented Aviary environments.

## Getting Started

To install `ldp`:

```bash
pip install -e .
```

To install `aviary` and the `nn` (neural network) module required for the tutorials:

```bash
pip install "ldp[nn]" "fhaviary[gsm8k]"
```

If you plan to export Graphviz visualizations, the `graphviz` library is required:

- Linux: `apt install graphviz`
- macOS: `brew install graphviz`

## Tutorial Notebooks

1. [Creating a Simple Language Agent][1]
2. [Evaluating a Llama Agent on GSM8K][2]

[1]: https://github.com/Future-House/ldp/blob/main/tutorials/creating_a_language_agent.ipynb
[2]: https://github.com/Future-House/ldp/blob/main/tutorials/evaluating_a_llama_agent.ipynb

## Running an Agent on an Aviary Environment

The minimal example below illustrates how to run a language agent on an Aviary environment
(LDP's sister library for defining language agent environments - <https://github.com/Future-House/aviary>)

```py
from ldp.agent import SimpleAgent
from aviary.core import DummyEnv

env = DummyEnv()
agent = SimpleAgent()

obs, tools = await env.reset()
agent_state = await agent.init_state(tools=tools)

done = False
while not done:
    action, agent_state, _ = await agent.get_asv(agent_state, obs)
    obs, reward, done, truncated = await env.step(action.value)
```

Below we elaborate on the components of LDP.

## Agent

An agent is a language agent that interacts with an environment to accomplish a task.
Agents may use tools (calls to external APIs e.g. Wolfram Alpha)
in response to observations returned by the environment.
Below we define LDP's `SimpleAgent` which relies on a single LLM call.
The main bookkeeping involves appending messages received from the environment and passing tools.

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

An agent has two methods:

```py
agent_state = await agent.init_state(tools=tools)
new_action, new_agent_state, value = await agent.get_asv(agent_state, obs)
```

- The `get_asv(agent_state, obs)` method chooses an action (`a`) conditioned on the observation messages
  returning the next agent state (`s`) and a value estimate (`v`).
- The first argument, `agent_state`, is an optional container for environment-specific objects such as
  e.g. documents for PaperQA or lookup results for HotpotQA,
- as well as more general objects such as memories which could include a list of previous actions and observations.
  `agent_state` may be set to `None` if memories are not being used.
- The second argument `obs` is not the complete list of all prior observations,
  but rather the returned value from `env.step`.
- The `value` is the agent's state/action value estimate used for reinforcment learning training. It may default to 0.

## A plain python agent

Want to just run python code? No problem - here's a minimal example of an Agent that is deterministic:

```py
from aviary.core import Message, Tool, ToolCall, ToolRequestMessage
from ldp.agent import Agent


class NoThinkAgent(Agent):
    async def init_state(self, tools):
        return None

    async def get_asv(self, tools, obs):
        tool_call = ToolCall.from_name("specific_tool_call", arg1="foo")
        action = ToolRequestMessage(tool_calls=[tool_call])
        return await Agent.wrap_action(action), None, 0.0
```

This agent has a state of `None`, just makes one specific tool call with `arg1="foo"`,
and then converts that into an action.
The only "magic" line of code is the `wrap_action`,
which just converts the action constructed by plain python into a node in a compute graph - see more below.

## Stochastic Computation Graph (SCG)

For more advanced use-cases, LDP features a stochastic computation graph [^2]
which enables differentiatiation with respect to agent parameters
(including the weights of the LLM).

You should install the `scg` subpackage to work with it:

```bash
pip install ldp[scg]
```

The example computation graph below illustrates the functionality

```py
from ldp.graph import FxnOp, LLMCallOp, PromptOp, compute_graph

op_a = FxnOp(lambda x: 2 * x)

async with compute_graph():
    op_result = op_a(3)
```

The code cell above creates and executes a computation graph that doubles the input.
The computation graph gradients and executions are saved in a context for later use, such as in training updates.
For example:

```py
print(op_result.compute_grads())
```

A more complex example is given below for an agent that possesses memory.

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

We use differentiable ops to ensure there is an edge in the compute graph from the LLM result (action)
to components such as memory retrieval as well as the query used to retrieve the memory.

Why use an SCG? Aside from the ability to take gradients,
using the SCG enables tracking of all inputs/outputs to the ops
and serialization/deserialization of the SCG such that it can be easily saved and loaded.
Input/output tracking also makes it easier to perform fine-tuning or reinforcement learning on the underlying LLMs.

## Generic Support

The `Agent` (as well as classes in `agent.ops`)
are [generics](https://en.wikipedia.org/wiki/Generic_programming),
which means:

- `Agent` is designed to support arbitrary types
- Subclasses can precisely specify state types, making the code more readable

If you are new to Python generics (`typing.Generic`),
please read about them in [Python `typing`](https://docs.python.org/3/library/typing.html#generics).
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

## References

[^1]: Narayanan, S., Braza, J.D., Griffiths, R.R., Ponnapati, M., Bou, A., Laurent, J., Kabeli, O., Wellawatte, G., Cox, S., Rodriques, S.G. and White, A.D., 2024. [Aviary: training language agents on challenging scientific tasks.](https://arxiv.org/abs/2412.21154) arXiv preprint arXiv:2412.21154.

[^2]: Schulman, J., Heess, N., Weber, T. and Abbeel, P., 2015. [Gradient estimation using stochastic computation graphs.](https://proceedings.neurips.cc/paper_files/paper/2015/hash/de03beffeed9da5f3639a621bcab5dd4-Abstract.html) Advances in Neural Information Processing Systems, 28.
