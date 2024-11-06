import litellm
from pydantic import BaseModel, ValidationError

from llmclient.model import LLMModel as BaseLLMModel
from llmclient.result import LLMResult

class JSONSchemaValidationError(ValueError):
    """Raised when the completion does not match the specified schema."""

def sum_logprobs(choice: litellm.utils.Choices) -> float | None:
    """Calculate the sum of the log probabilities of an LLM completion (a Choices object).

    Args:
        choice: A sequence of choices from the completion.

    Returns:
        The sum of the log probabilities of the choice.
    """
    try:
        logprob_obj = choice.logprobs
    except AttributeError:
        return None
    if isinstance(logprob_obj, dict):
        if logprob_obj.get("content"):
            return sum(
                logprob_info["logprob"] for logprob_info in logprob_obj["content"]
            )
    elif choice.logprobs.content:
        return sum(logprob_info.logprob for logprob_info in choice.logprobs.content)
    return None


def validate_json_completion(
    completion: litellm.ModelResponse, output_type: type[BaseModel]
) -> None:
    """Validate a completion against a JSON schema.

    Args:
        completion: The completion to validate.
        output_type: The Pydantic model to validate the completion against.
    """
    try:
        for choice in completion.choices:
            if not hasattr(choice, "message") or not choice.message.content:
                continue
            # make sure it is a JSON completion, even if None
            # We do want to modify the underlying message
            # so that users of it can just parse it as expected
            choice.message.content = (
                choice.message.content.split("```json")[-1].split("```")[0] or ""
            )
            output_type.model_validate_json(choice.message.content)
    except ValidationError as err:
        raise JSONSchemaValidationError(
            "The completion does not match the specified schema."
        ) from err


class LLMModel(BaseLLMModel):
    async def call(self, *args, **kwargs) -> LLMResult:  # type: ignore[override]
        return (await super().call(*args, **kwargs))[0]
