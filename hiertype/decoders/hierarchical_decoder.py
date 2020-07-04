from typing import *
from abc import ABC, abstractmethod
import torch

from hiertype.data import Hierarchy


class HierarchyDecoder(ABC):

    def __init__(self, hierarchy: Hierarchy):
        self.hierarchy = hierarchy

    @abstractmethod
    def decode(self,
               y: torch.Tensor  # R[Batch, Type]
               ) -> List[Set[int]]:
        pass
