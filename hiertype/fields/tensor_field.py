from typing import *
import torch
from allennlp.data import DataArray

from allennlp.data.fields import Field


class TensorField(Field[torch.Tensor]):
    """
    A field that contains tensors (from other input sources, e.g. HDF5).
    A tensor could have 1 dimension that is the sequence length:
    When being batched, that dimension will automatically be padded.
    """

    def __init__(self, tensor: torch.Tensor, pad_dim: int, pad_element=0):
        self.tensor = tensor
        self.pad_dim = pad_dim
        self.pad_element = pad_element

    def get_padding_lengths(self) -> Dict[str, int]:
        return {"length": self.tensor.size(self.pad_dim)}

    def as_tensor(self, padding_lengths: Dict[str, int]) -> DataArray:
        pad_shape = list(self.tensor.size())
        pad_shape[self.pad_dim] = padding_lengths["length"] - self.tensor.size(self.pad_dim)

        pad = torch.full(pad_shape, self.pad_element).type_as(self.tensor)
        return torch.cat([self.tensor, pad], dim=self.pad_dim)

    def empty_field(self) -> 'Field':
        raise NotImplementedError

    def __str__(self) -> str:
        return f"TensorField with shape: {self.tensor.size()}"
