import contextvars
import logging
from contextlib import contextmanager
from datetime import datetime
from typing import TypeAlias
from uuid import UUID, uuid4

import litellm
from aviary.core import Message
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
)

logger = logging.getLogger(__name__)

# A context var that will be unique to threads/processes
cvar_session_id = contextvars.ContextVar[UUID | None]("session_id", default=None)

# Type alias for LLM response types that can be tracked for cost
LLMResponse: TypeAlias = (
    litellm.ModelResponse | litellm.EmbeddingResponse | litellm.ModelResponseStream
)


@contextmanager
def set_llm_session_ids(session_id: UUID):
    token = cvar_session_id.set(session_id)
    try:
        yield
    finally:
        cvar_session_id.reset(token)


class Embeddable(BaseModel):
    embedding: list[float] | None = Field(default=None, repr=False)


class LLMResult(BaseModel):
    """A class to hold the result of a LLM completion.

    To associate a group of LLMResults, you can use the `set_llm_session_ids` context manager:

    ```python
    my_session_id = uuid4()
    with set_llm_session_ids(my_session_id):
        # code that generates LLMResults
        pass
    ```

    and all the LLMResults generated within the context will have the same `session_id`.
    This can be combined with LLMModels `llm_result_callback` to store all LLMResults.
    """

    model_config = ConfigDict(populate_by_name=True)

    id: UUID = Field(default_factory=uuid4)
    session_id: UUID | None = Field(
        default_factory=cvar_session_id.get,  # type: ignore[arg-type]
        description="A persistent ID to associate a group of LLMResults",
        alias="answer_id",
    )
    name: str | None = None
    config: dict | None = None
    prompt: str | list[dict] | Message | list[Message] | None = Field(
        default=None,
        description="Optional prompt or list of serialized prompts.",
    )
    text: str | None = None
    messages: list[Message] | None = Field(
        default=None, description="Messages received from the LLM."
    )
    prompt_count: int = 0
    completion_count: int = 0
    model: str
    date: str = Field(default_factory=datetime.now().isoformat)

    # Cached token counts (normalized across providers)
    # - Anthropic: cache_read_input_tokens, cache_creation_input_tokens
    # - OpenAI: prompt_tokens_details.cached_tokens (read only)
    cache_read_tokens: int = Field(
        default=0,
        description="Tokens read from cache (Anthropic/OpenAI). Default 0 means no cache hits.",
    )
    cache_creation_tokens: int = Field(
        default=0,
        description="Tokens written to cache (Anthropic only). Default 0 means no cache creation.",
    )

    # Store accurate cost from litellm (handles all pricing tiers and provider quirks)
    cost_usd: float | None = Field(
        default=None,
        description="Calculated by litellm.completion_cost(). Accurate for cached tokens.",
    )
    seconds_to_first_token: float = Field(
        default=0.0, description="Delta time (sec) to first response token's arrival."
    )
    seconds_to_last_token: float = Field(
        default=0.0, description="Delta time (sec) to last response token's arrival."
    )
    logprob: float | None = Field(
        default=None, description="Sum of logprobs in the completion."
    )
    top_logprobs: list[list[tuple[str, float]]] | None = Field(
        default=None, description="Top logprobs for each position in the completion."
    )
    reasoning_content: str | None = Field(
        default=None, description="Reasoning content from LLMs such as DeepSeek-R1."
    )

    def __str__(self) -> str:
        return self.text or ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cost(self) -> float:
        """Return the cost of the result in dollars.

        Uses litellm.completion_cost() result if available (accurate for cached tokens).
        Falls back to simple calculation (inaccurate for cached tokens).
        """
        # Prefer the accurate cost from litellm
        if self.cost_usd is not None:
            return self.cost_usd

        # Fallback: simple calculation (DOES NOT account for cached tokens)
        if self.prompt_count and self.completion_count:
            try:
                pc = litellm.model_cost[self.model]["input_cost_per_token"]
                oc = litellm.model_cost[self.model]["output_cost_per_token"]
                return pc * self.prompt_count + oc * self.completion_count
            except KeyError:
                logger.warning(f"Could not find cost for model {self.model}.")
        return 0.0

    # TODO: These two methods were implemented in ldp, but not in pqa.
    # TODO: Check if they're necessary
    @property
    def provider(self) -> str:
        """Get the model provider's name (e.g. "openai", "mistral")."""
        return litellm.get_llm_provider(self.model)[1]

    def get_supported_openai_params(self) -> list[str] | None:
        """Get the supported OpenAI parameters for the model."""
        return litellm.get_supported_openai_params(self.model)
