from typing import Any


class JSONSchemaValidationError(ValueError):
    """Raised when the completion does not match the specified schema."""


class ModelRefusalError(RuntimeError):
    """Raised when an LLM declines to complete a request (e.g. content filter).

    Carries the raw provider response so callers that choose to handle the
    refusal (rather than fall back) can still inspect it.
    """

    def __init__(
        self,
        message: str,
        *,
        model: str,
        finish_reason: str | None,
        response: Any = None,
    ) -> None:
        super().__init__(message)
        self.model = model
        self.finish_reason = finish_reason
        self.response = response


class ResponseValidationError(RuntimeError):
    """Raised when an `LLMConfig.response_validator` rejects an `LLMResult`.

    Treated as transient by the retry/fallback loop so the validator gets a
    fresh attempt at the same model (up to `ModelSpec.max_retries`) before
    advancing to the next model.
    """


class AllModelsExhaustedError(RuntimeError):
    """Raised when every model in an `LLMConfig.models` chain has failed or been skipped."""

    def __init__(self, last_exc: BaseException | None = None) -> None:
        super().__init__(
            "All models in the LLMConfig chain failed."
            + (f" Last error: {last_exc!r}" if last_exc is not None else "")
        )
        self.last_exc = last_exc
