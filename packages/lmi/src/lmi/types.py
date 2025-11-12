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
    prompt_count: int | None = Field(default=None, ge=0)
    completion_count: int | None = Field(default=None, ge=0)
    model: str
    date: str = Field(default_factory=datetime.now().isoformat)

    # Cached token counts - extracted from provider-specific usage fields:
    # - Both providers report cache reads via prompt_tokens_details.cached_tokens
    # - Only Anthropic reports cache creation via cache_creation_input_tokens
    cache_read_tokens: int | None = Field(
        default=None,
        frozen=True,
        ge=0,
        description=(
            "Tokens read from cache (Anthropic/OpenAI). "
            "None means caching wasn't used, 0 means caching was used but no cache hits."
        ),
    )
    cache_creation_tokens: int | None = Field(
        default=None,
        frozen=True,
        ge=0,
        description=(
            "Tokens written to cache (Anthropic only). "
            "None means caching wasn't used, 0 means caching was used but no cache creation."
        ),
    )

    cost: float = Field(
        default=0.0,
        frozen=True,
        ge=0,
        description="Cost (USD).",
    )
    seconds_to_first_token: float = Field(
        default=0.0,
        ge=0,
        description="Delta time (sec) to first response token's arrival.",
    )
    seconds_to_last_token: float = Field(
        default=0.0,
        ge=0,
        description="Delta time (sec) to last response token's arrival.",
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

    # TODO: These two methods were implemented in ldp, but not in pqa.
    # TODO: Check if they're necessary
    @property
    def provider(self) -> str:
        """Get the model provider's name (e.g. "openai", "mistral")."""
        return litellm.get_llm_provider(self.model)[1]

    def get_supported_openai_params(self) -> list[str] | None:
        """Get the supported OpenAI parameters for the model."""
        return litellm.get_supported_openai_params(self.model)
