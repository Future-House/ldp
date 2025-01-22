from abc import ABC, abstractmethod

import torch
from transformers import LogitsProcessor


class LogitsProcessorWithFinalize(LogitsProcessor, ABC):
    @abstractmethod
    def finalize(self, token_ids: torch.Tensor) -> None:
        pass
