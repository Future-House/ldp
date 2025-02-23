from sys import version_info

import litellm

# Estimate from OpenAI's FAQ
# https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
CHARACTERS_PER_TOKEN_ASSUMPTION: float = 4.0
# Added tokens from user/role message
# Need to add while doing rate limits
# Taken from empirical counts in tests
EXTRA_TOKENS_FROM_USER_ROLE: int = 7

MODEL_COST_MAP = litellm.get_model_cost_map("")

DEFAULT_VERTEX_SAFETY_SETTINGS: list[dict[str, str]] = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_ONLY_HIGH",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_ONLY_HIGH",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_ONLY_HIGH",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_ONLY_HIGH",
    },
]

IS_PYTHON_BELOW_312 = version_info < (3, 12)
