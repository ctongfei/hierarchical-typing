from typing import *


class Alphabet:
    """
    Maintains a bijection between symbols (tokens / labels) with unique indices.
    """

    def __init__(self,
                 sym_to_idx: Dict[str, int],
                 idx_to_sym: List[str]
                 ):
        self.sym_to_idx: Dict[str, int] = sym_to_idx
        self.idx_to_sym: List[str] = idx_to_sym

    def size(self) -> int:
        return len(self.idx_to_sym)

    def index(self, sym: str) -> int:
        if sym in self.sym_to_idx:
            return self.sym_to_idx[sym]
        else:
            idx = self.size()
            self.idx_to_sym.append(sym)
            self.sym_to_idx[sym] = idx
            return idx

    @classmethod
    def with_special_symbols(cls, special_symbols: List[str]) -> 'Alphabet':

        sym_to_idx: Dict[str, int] = {}
        idx_to_sym: List[str] = []

        for i, sym in enumerate(special_symbols):
            sym_to_idx[sym] = i
            idx_to_sym.append(sym)

        return cls(sym_to_idx, idx_to_sym)
