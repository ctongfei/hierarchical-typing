from typing import *
import torch
import math

from allennlp.data.fields import Field


class RealField(Field[torch.Tensor]):
    """
    A `RealField` contains a real-valued number.
    This field will be converted into a batched float tensor.
    """

    def __init__(self, value: float):
        self.value = value

    def get_padding_lengths(self) -> Dict[str, int]:
        return {}

    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        return torch.tensor(self.value, dtype=torch.float32)

    def empty_field(self) -> 'Field':
        return RealField(math.nan)

    def __str__(self) -> str:
        return f"RealField with value: {self.value}"
    