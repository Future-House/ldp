[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg?style=plastic)]()
[![tests](https://github.com/Future-House/llm-client/actions/workflows/test.yaml/badge.svg?style=plastic)](https://github.com/Future-House/ldp/tree/main/packages/llmi)
[![PyPI version](https://badge.fury.io/py/lmi.svg?style=plastic)](https://badge.fury.io/py/lmi)

# Language Model Interface (LMI)

A Python library for interacting with Large Language Models (LLMs) through an unified interface.

## Installation

```bash
pip install lmi
```

<!--TOC-->

- [Language Model Interface (LMI)](#language-model-interface-lmi)
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
    - [Embedding models](#embedding-models)
      - [LiteLLMEmbeddingModel](#litellmembeddingmodel)
      - [HybridEmbeddingModel](#hybridembeddingmodel)
      - [SentenceTransformerEmbeddingModel](#sentencetransformerembeddingmodel)

<!--TOC-->

## Quick start

A simple example of how to use the library with default settings is shown below.

```python
from lmi import LiteLLMModel
from aviary import Message

llm = LiteLLMModel()

messages = [Message(content="What is the meaning of life?")]

completion = await llm.call_single(messages)
assert completion.text == "42"
```

## Documentation

### LLMs

An LLM is a class that inherits from `LLMModel` and implements the following methods:

- `async acompletion(messages: list[Message], **kwargs) -> list[LLMResult]`
- `async acompletion_iter(messages: list[Message], **kwargs) -> AsyncIterator[LLMResult]`

These methods are used by the base class `LLMModel` to implement the LLM interface.
Because `LLMModel` is an abstract class, it doesn't depend on any specific LLM provider. All the connection with the provider is done in the subclasses using `acompletion` and `acompletion_iter` as interfaces.

Because these are the only methods that communicate with the chosen LLM provider, we use an abstraction [LLMResult](https://github.com/Future-House/ldp/blob/main/packages/lmi/src/lmi/types.py#L35) to hold the results of the LLM call.

#### LLMModel

An `LLMModel` implements `call`, which receives a list of `aviary.Message`s and returns a list of `LLMResult`s. `LLMModel.call` can receive callbacks, tools, and output schemas to control its behavior, as better explained below.
Adittionally, `LLMModel.call_single` can be used to return a single `LLMResult` completion.

#### LiteLLMModel

`LiteLLMModel` wraps `LiteLLM` API usage within our `LLMModel` interface. It receives a `name` parameter, which is the name of the model to use and a `config` parameter, which is a dictionary of configuration options for the model following the [LiteLLM configuration schema](https://docs.litellm.ai/docs/routing). Common parameters such as `temperature`, `max_token`, and `n` (the number of completions to return) can be passed as part of the `config` dictionary.

```python
from lmi import LiteLLMModel

config = {
    "model_list": [
        {
            "model_name": "gpt-4o",
            "litellm_params": {
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
config = {
    {
        "name": "gpt-4o",
        "temperature": 0.1,
        "max_tokens": 512,
        "n": 5,
    }
}

llm = LiteLLMModel(config=config)
```

### Cost tracking

Cost tracking is supported in two different ways:

1. Calls to the LLM returns the token usage for each call in `LLMResult.prompt_count` and `LLMResult.completion_count`. Additionally, `LLMResult.cost` can be used to get a cost estimate for the call in USD.
2. A global cost tracker is maintained in `GLOBAL_COST_TRACKER` and can be enabled or disabled using `enable_cost_tracking()` and `cost_tracking_ctx()`.

### Rate limiting

Rate limiting helps control the rate of requests made to various services and LLMs. The rate limiter supports both in-memory and Redis-based storage for cross-process rate limiting.

#### Basic Usage

Rate limits can be configured in two ways:

1. Through the LLM configuration:

```python
from lmi import LiteLLMModel

config = {
    "rate_limit": {
        "gpt-4": "100/minute",  # 100 tokens per minute
    }
}

llm = LiteLLMModel(name="gpt-4", config=config)
```

2. Through the global rate limiter configuration:

```python
from lmi.rate_limiter import GLOBAL_LIMITER

GLOBAL_LIMITER.rate_config[("client", "gpt-4")] = "100/minute"
```

#### Rate Limit Format

Rate limits can be specified in two formats:

1. As a string: `"<count> [per|/] [n (optional)] <second|minute|hour|day|month|year>"`

   ```python
   "100/minute"  # 100 requests per minute

   "5 per second"  # 5 requests per second
   "1000/day"  # 1000 requests per day
   ```

2. Using RateLimitItem classes:

   ```python
   from limits import RateLimitItemPerSecond, RateLimitItemPerMinute

   RateLimitItemPerSecond(30, 1)  # 30 requests per second
   RateLimitItemPerMinute(1000, 1)  # 1000 requests per minute
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

#### Monitoring Rate Limits

You can monitor current rate limit status:

```python
from lmi.rate_limiter import GLOBAL_LIMITER

status = await GLOBAL_LIMITER.rate_limit_status()

# Example output:
{
    ("client", "gpt-4"): {
        "period_start": 1234567890,
        "n_items_in_period": 50,
        "period_seconds": 60,
        "period_name": "minute",
        "period_cap": 100,
    }
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

🚧 [ WIP ] 🚧

### Embedding models

This client also includes embedding models. An embedding model is a class that inherits from `EmbeddingModel` and implements the `embed_documents` method, which receives a list of strings and returns a list with a list of floats (the embeddings) for each string.

Currently, the following embedding models are supported:

- `LiteLLMEmbeddingModel`
- `SparseEmbeddingModel`
- `SentenceTransformerEmbeddingModel`
- `HybridEmbeddingModel`

#### LiteLLMEmbeddingModel

`LiteLLMEmbeddingModel` provides a wrapper around LiteLLM's embedding functionality. It supports various embedding models through the LiteLLM interface, with automatic dimension inference and token limit handling. It defaults to `text-embedding-3-small` and can be configured with a `name`, `batch_size`, and `config` parameters.
Notice that `LiteLLMEmbeddingModel` can also be rate limited.

```python
from lmi import LiteLLMEmbeddingModel

model = LiteLLMEmbeddingModel()

model = LiteLLMEmbeddingModel(
    name="text-embedding-ada-002",
    batch_size=16,
    config={
        "kwargs": {
            "api_key": "your-api-key",  # pragma: allowlist secret
        },
        "rate_limit": "100/minute",
    },
)

embeddings = await model.embed_documents(["text1", "text2", "text3"])
```

#### HybridEmbeddingModel

`HybridEmbeddingModel` combines multiple embedding models by concatenating their outputs. It is typically used to combine a dense embedding model (like `LiteLLMEmbeddingModel`) with a sparse embedding model for improved performance. The model can be created in two ways:

```python
from lmi import LiteLLMEmbeddingModel, SparseEmbeddingModel, HybridEmbeddingModel

dense_model = LiteLLMEmbeddingModel(name="text-embedding-3-small")
sparse_model = SparseEmbeddingModel()
hybrid_model = HybridEmbeddingModel(models=[dense_model, sparse_model])
```

The resulting embedding dimension will be the sum of the dimensions of all component models. For example, if you combine a 1536-dimensional dense embedding with a 256-dimensional sparse embedding, the final embedding will be 1792-dimensional.

#### SentenceTransformerEmbeddingModel

You can also use `sentence-transformer`, which is a local embedding library with support for HuggingFace models, by installing `lmi[local]`.
