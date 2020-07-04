from typing import *
import torch
import math

from allennlp.data.fields import Field


class IntField(Field[torch.Tensor]):
    """
    An `IntField` contains a real-valued number.
    This field will be converted into a batched long tensor.
    This is different than an `allennlp.data.fields.LabelField`, where the semantics is categorical.
    """

    def __init__(self, value: int):
        self.value = value

    def get_padding_lengths(self) -> Dict[str, int]:
        return {}

    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        return torch.tensor(self.value, dtype=torch.int64)

    def empty_field(self) -> 'Field':
        return IntField(0)

    def __str__(self) -> str:
        return f"IntField with value: {self.value}"