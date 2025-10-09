# Language Model Interface (LMI)

<!-- pyml disable-num-lines 6 line-length -->

[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Future-House/ldp/tree/main/packages/lmi)
[![PyPI version](https://badge.fury.io/py/fhlmi.svg)](https://badge.fury.io/py/fhlmi)
[![tests](https://github.com/Future-House/ldp/actions/workflows/tests.yml/badge.svg)](https://github.com/Future-House/ldp/tree/main/packages/lmi)
![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)
![PyPI Python Versions](https://img.shields.io/pypi/pyversions/fhlmi)

A Python library for interacting with Large Language Models (LLMs)
through a unified interface,
hence the name Language Model Interface (LMI).

## Installation

```bash
pip install fhlmi
```

<!--TOC-->

- [Installation](#installation)
- [Quick start](#quick-start)
- [Documentation](#documentation)
  - [LLMs](#llms)
    - [LLMModel](#llmmodel)
    - [LiteLLMModel](#litellmmodel)
  - [Cost tracking](#cost-tracking)
  - [Rate limiting](#rate-limiting)
    - [Basic Usage](#basic-usage)
    - [Rate Limit Format](#rate-limit-format)
    - [Storage Options](#storage-options)
    - [Monitoring Rate Limits](#monitoring-rate-limits)
    - [Timeout Configuration](#timeout-configuration)
    - [Weight-based Rate Limiting](#weight-based-rate-limiting)
  - [Tool calling](#tool-calling)
  - [Vertex](#vertex)
  - [Embedding models](#embedding-models)
    - [LiteLLMEmbeddingModel](#litellmembeddingmodel)
    - [HybridEmbeddingModel](#hybridembeddingmodel)
    - [SentenceTransformerEmbeddingModel](#sentencetransformerembeddingmodel)

<!--TOC-->

## Quick start

A simple example of how to use the library with default settings is shown below.

```python
from lmi import LiteLLMModel
from aviary.core import Message

llm = LiteLLMModel()

messages = [Message(content="What is the meaning of life?")]

result = await llm.call_single(messages)
# assert result.text == "42"
```

or, if you only have one user message, just:

```python
from lmi import LiteLLMModel

llm = LiteLLMModel()
result = await llm.call_single("What is the meaning of life?")
# assert result.text == "42"
```

## Documentation

### LLMs

An LLM is a class that inherits from `LLMModel` and implements the following methods:

- `async acompletion(messages: list[Message], **kwargs) -> list[LLMResult]`
- `async acompletion_iter(messages: list[Message], **kwargs) -> AsyncIterator[LLMResult]`

These methods are used by the base class `LLMModel` to implement the LLM interface.
Because `LLMModel` is an abstract class, it doesn't depend on any specific LLM provider.
All the connection with the provider is done in the subclasses using `acompletion` and `acompletion_iter` as interfaces.

Because these are the only methods that communicate with the chosen LLM provider,
we use an abstraction [LLMResult](https://github.com/Future-House/ldp/blob/main/packages/lmi/src/lmi/types.py#L35)
to hold the results of the LLM call.

#### LLMModel

An `LLMModel` implements `call`, which receives a list of `aviary` `Message`s and returns a list of `LLMResult`s.
`LLMModel.call` can receive callbacks, tools, and output schemas to control its behavior, as better explained below.
Because we support interacting with the LLMs using `Message` objects, we can use the modalities available in `aviary`,
which currently include text and images.
`lmi` supports these modalities but does not support other modalities yet.
Adittionally, `LLMModel.call_single` can be used to return a single `LLMResult` completion.

#### LiteLLMModel

`LiteLLMModel` wraps `LiteLLM` API usage within our `LLMModel` interface.
It receives a `name` parameter,
which is the name of the model to use and a `config` parameter,
which is a dictionary of configuration options for the model following the
[LiteLLM configuration schema](https://docs.litellm.ai/docs/routing).
Common parameters such as `temperature`, `max_token`, and `n` (the number of completions to return)
can be passed as part of the `config` dictionary.

```python
import os
from lmi import LiteLLMModel

config = {
    "model_list": [
        {
            "model_name": "gpt-4o",
            "litellm_params": {
                "model": "gpt-4o",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "frequency_penalty": 1.5,
                "top_p": 0.9,
                "max_tokens": 512,
                "temperature": 0.1,
                "n": 5,
            },
        }
    ]
}

llm = LiteLLMModel(name="gpt-4o", config=config)
```

`config` can also be used to pass common parameters directly for the model.

```python
from lmi import LiteLLMModel

config = {
    "name": "gpt-4o",
    "temperature": 0.1,
    "max_tokens": 512,
    "n": 5,
}

llm = LiteLLMModel(config=config)
```

### Cost tracking

Cost tracking is supported in two different ways:

1. Calls to the LLM return the token usage for each call in `LLMResult.prompt_count` and `LLMResult.completion_count`.
   Additionally, `LLMResult.cost` can be used to get a cost estimate for the call in USD.
2. A global cost tracker is maintained in `GLOBAL_COST_TRACKER`
   and can be enabled or disabled using `enable_cost_tracking()` and `cost_tracking_ctx()`.

### Rate limiting

Rate limiting helps regulate the usage of resources to various services and LLMs.
The rate limiter supports both in-memory and Redis-based storage for cross-process rate limiting.
Currently, `lmi` take into account the tokens used (Tokens per Minute (TPM))
and the requests handled (Requests per Minute (RPM)).

#### Basic Usage

Rate limits can be configured in two ways:

1. Through the LLM configuration:

   ```python
   from lmi import LiteLLMModel

   config = {
       "rate_limit": {
           "gpt-4": "100/minute",  # 100 tokens per minute
       },
       "request_limit": {
           "gpt-4": "5/minute",  # 5 requests per minute
       },
   }

   llm = LiteLLMModel(name="gpt-4", config=config)
   ```

   With `rate_limit` we rate limit only token consumption,
   and with `request_limit` we rate limit only request volume.
   You can configure both of them or only one of them as you need.

2. Through the global rate limiter configuration:

   ```python
   from lmi.rate_limiter import GLOBAL_LIMITER

   GLOBAL_LIMITER.rate_config[("client", "gpt-4")] = "100/minute"  # tokens per minute
   GLOBAL_LIMITER.rate_config[("client|request", "gpt-4")] = (
       "5/minute"  # requests per minute
   )
   ```

   With `client` we rate limit only token consumption,
   and with `client|request` we rate limit only request volume.
   You can configure both of them or only one of them as you need.

#### Rate Limit Format

Rate limits can be specified in two formats:

1. As a string: `"<count> [per|/] [n (optional)] <second|minute|hour|day|month|year>"`

   ```python
   "100/minute"  # 100 tokens per minute

   "5 per second"  # 5 tokens per second
   "1000/day"  # 1000 tokens per day
   ```

2. Using RateLimitItem classes:

   ```python
   from limits import RateLimitItemPerSecond, RateLimitItemPerMinute

   RateLimitItemPerSecond(30, 1)  # 30 tokens per second
   RateLimitItemPerMinute(1000, 1)  # 1000 tokens per minute
   ```

#### Storage Options

The rate limiter supports two storage backends:

1. In-memory storage (default when Redis is not configured):

   ```python
   from lmi.rate_limiter import GlobalRateLimiter

   limiter = GlobalRateLimiter(use_in_memory=True)
   ```

2. Redis storage (for cross-process rate limiting):

   ```python
   # Set REDIS_URL environment variable
   import os

   os.environ["REDIS_URL"] = "localhost:6379"

   from lmi.rate_limiter import GlobalRateLimiter

   limiter = GlobalRateLimiter()  # Will automatically use Redis if REDIS_URL is set
   ```

   This `limiter` can be used in within the `LLMModel.check_rate_limit` method
   to check the rate limit before making a request,
   similarly to how it is done in the [`LiteLLMModel` class][1].

[1]: https://github.com/Future-House/ldp/blob/18138af155bef7686d1eb2b486edbc02d62037eb/packages/lmi/src/lmi/llms.py

#### Monitoring Rate Limits

You can monitor current rate limit status:

```python
from lmi.rate_limiter import GLOBAL_LIMITER
from lmi import LiteLLMModel
from aviary.core import Message

config = {
    "rate_limit": {
        "gpt-4": "100/minute",  # 100 tokens per minute
    },
    "request_limit": {
        "gpt-4": "5/minute",  # 5 requests per minute
    },
}

llm = LiteLLMModel(name="gpt-4", config=config)
results = await llm.call([Message(content="Hello, world!")])  # Consume some tokens

status = await GLOBAL_LIMITER.rate_limit_status()

# Example output:
{
    ("client|request", "gpt-4"): {  # the limit status for requests
        "period_start": 1234567890,
        "n_items_in_period": 1,
        "period_seconds": 60,
        "period_name": "minute",
        "period_cap": 5,
    },
    ("client", "gpt-4"): {  # the limit status for tokens
        "period_start": 1234567890,
        "n_items_in_period": 50,
        "period_seconds": 60,
        "period_name": "minute",
        "period_cap": 100,
    },
}
```

#### Timeout Configuration

The default timeout for rate limiting is 60 seconds, but can be configured:

```python
import os

os.environ["RATE_LIMITER_TIMEOUT"] = "30"  # 30 seconds timeout
```

#### Weight-based Rate Limiting

Rate limits can account for different weights (e.g., token counts for LLM requests):

```python
await GLOBAL_LIMITER.try_acquire(
    ("client", "gpt-4"),
    weight=token_count,  # Number of tokens in the request
    acquire_timeout=30.0,  # Optional timeout override
)
```

### Tool calling

LMI supports function calling through tools, which are functions that the LLM can invoke.
Tools are passed to `LLMModel.call` or `LLMModel.call_single`
as a list of [`Tool` objects from `aviary`][2],
along with an optional `tool_choice` parameter that controls how the LLM uses these tools.

[2]: https://github.com/Future-House/aviary/blob/1a50b116fb317c3ef27b45ea628781eb53c0b7ae/src/aviary/tools/base.py#L334

The `tool_choice` parameter follows `OpenAI`'s definition. It can be:

| Tool Choice Value               | Constant                           | Behavior                                                                       |
| ------------------------------- | ---------------------------------- | ------------------------------------------------------------------------------ |
| `"none"`                        | `LLMModel.NO_TOOL_CHOICE`          | The model will not call any tools and instead generates a message              |
| `"auto"`                        | `LLMModel.MODEL_CHOOSES_TOOL`      | The model can choose between generating a message or calling one or more tools |
| `"required"`                    | `LLMModel.TOOL_CHOICE_REQUIRED`    | The model must call one or more tools                                          |
| A specific `aviary.Tool` object | N/A                                | The model must call this specific tool                                         |
| `None`                          | `LLMModel.UNSPECIFIED_TOOL_CHOICE` | No tool choice preference is provided to the LLM API                           |

When tools are provided, the LLM's response will be wrapped in a `ToolRequestMessage` instead of a regular `Message`.
The key differences are:

- `Message` represents a basic chat message with a role (system/user/assistant) and content
- `ToolRequestMessage` extends `Message` to include `tool_calls`, which contains a list of `ToolCall` objects,
  which contains the tools the LLM chose to invoke and their arguments

Further details about how to define a tool,
use the `ToolRequestMessage` and the `ToolCall` objects can be found in the
[Aviary documentation](https://github.com/Future-House/aviary?tab=readme-ov-file#tool).

Here is a minimal example usage:

```python
from lmi import LiteLLMModel
from aviary.core import Message, Tool
import operator


# Define a function that will be used as a tool
def calculator(operation: str, x: float, y: float) -> float:
    """
    Performs basic arithmetic operations on two numbers.

    Args:
        operation (str): The arithmetic operation to perform ("+", "-", "*", or "/")
        x (float): The first number
        y (float): The second number

    Returns:
        float: The result of applying the operation to x and y

    Raises:
        KeyError: If operation is not one of "+", "-", "*", "/"
        ZeroDivisionError: If operation is "/" and y is 0
    """
    operations = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
    }
    return operations[operation](x, y)


# Create a tool from the calculator function
calculator_tool = Tool.from_function(calculator)

# The LLM must use the calculator tool
llm = LiteLLMModel()
result = await llm.call_single(
    messages=[Message(content="What is 2 + 2?")],
    tools=[calculator_tool],
    tool_choice=LiteLLMModel.TOOL_CHOICE_REQUIRED,
)

# result.messages[0] will be a ToolRequestMessage with tool_calls containing
# the calculator invocation with x=2, y=2, operation="+"
```

### Vertex

Vertex requires a bit of extra set-up. First, install the extra dependency for auth:

```sh
pip install google-api-python-client
```

and then you need to configure which region/project you're using for the model calls.
Make sure you're authed for that region/project. Typically that means running:

```sh
gcloud auth application-default login
```

Then you can use vertex models:

```py
from lmi import LiteLLMModel
from aviary.core import Message

vertex_config = {"vertex_project": "PROJECT_ID", "vertex_location": "REGION"}

llm = LiteLLMModel(name="vertex_ai/gemini-2.5-pro", config=vertex_config)
await llm.call_single("hey")
```

### Embedding models

This client also includes embedding models.
An embedding model is a class that inherits from `EmbeddingModel`
and implements the `embed_documents` method,
which receives a list of strings
and returns a list with a list of floats (the embeddings) for each string.

Currently, the following embedding models are supported:

- `LiteLLMEmbeddingModel`
- `SparseEmbeddingModel`
- `SentenceTransformerEmbeddingModel`
- `HybridEmbeddingModel`

#### LiteLLMEmbeddingModel

`LiteLLMEmbeddingModel` provides a wrapper around LiteLLM's embedding functionality.
It supports various embedding models through the LiteLLM interface,
with automatic dimension inference and token limit handling.
It defaults to `text-embedding-3-small` and can be configured with `name` and `config` parameters.
Notice that `LiteLLMEmbeddingModel` can also be rate limited.

```python
from lmi import LiteLLMEmbeddingModel

model = LiteLLMEmbeddingModel(
    name="text-embedding-3-small",
    config={"rate_limit": "100/minute", "batch_size": 16},
)

embeddings = await model.embed_documents(["text1", "text2", "text3"])
```

#### HybridEmbeddingModel

`HybridEmbeddingModel` combines multiple embedding models by concatenating their outputs.
It is typically used to combine a dense embedding model (like `LiteLLMEmbeddingModel`)
with a sparse embedding model for improved performance.
The model can be created in two ways:

```python
from lmi import LiteLLMEmbeddingModel, SparseEmbeddingModel, HybridEmbeddingModel

dense_model = LiteLLMEmbeddingModel(name="text-embedding-3-small")
sparse_model = SparseEmbeddingModel()
hybrid_model = HybridEmbeddingModel(models=[dense_model, sparse_model])
```

The resulting embedding dimension will be the sum of the dimensions of all component models.
For example, if you combine a 1536-dimensional dense embedding with a 256-dimensional sparse embedding,
the final embedding will be 1792-dimensional.

#### SentenceTransformerEmbeddingModel

You can also use `sentence-transformer`,
which is a local embedding library with support for HuggingFace models,
by installing `lmi[local]`.
