from abc import ABC, abstractmethod

import torch
from transformers import LogitsProcessor


class LogitsProcessorWithFinalize(LogitsProcessor, ABC):
    @abstractmethod
    def finalize(self, token_ids: torch.Tensor) -> None:
        """A method for subclasses to inject arbitrary finalization logic after sampling finishes.

        TransformerHandler will invoke logit_processor.finalize(token_ids), where token_ids are
        the sampled tokens.
        """
