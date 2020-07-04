from typing import *
import torch

from hiertype.data import Hierarchy
from hiertype.decoders import HierarchyDecoder


class BeamDecoder(HierarchyDecoder):

    def __init__(self,
                 hierarchy: Hierarchy,
                 strategies: List[str],  # top / other
                 max_branching_factors: List[int],
                 delta: List[float],
                 top_other_delta: float = 0.0
                 ):
        super(BeamDecoder, self).__init__(hierarchy)
        self.max_branching_factors = max_branching_factors
        self.delta = delta
        self.strategies = strategies
        self.top_other_delta = top_other_delta

    def decode(self,
               predictions: torch.Tensor  # R[Batch, Type]
               ) -> List[Set[int]]:

        def step_weak(y: torch.Tensor, l: int) -> Set[int]:  # decode without adhering to the hierarchy
            threshold = y[0].item() + self.delta[l]  # 0: root threshold
            children = [(s, y[s].item()) for s in self.hierarchy.level_range(l)]
            children.sort(key=lambda p: p[1], reverse=True)

            accepted_children: List[Tuple[int, float]] = []
            for i in range(min(self.max_branching_factors[l], len(children))):
                s, ys = children[i]
                if ys < threshold:
                    break
                accepted_children.append(children[i])
            return {s for s, _ in accepted_children}

        def step(y: torch.Tensor,  # R[Type]
                 l: int,
                 t: int) -> Tuple[Set[int], bool]:

            threshold = y[t].item() + self.delta[l]
            if self.hierarchy.type_str(t) == '/other':
                threshold += self.top_other_delta

            children: List[Tuple[int, float]] = [(s, y[s].item()) for s in self.hierarchy.children(t)]
            children.sort(key=lambda p: p[1], reverse=True)

            accepted_children: List[Tuple[int, float]] = []

            for i in range(min(self.max_branching_factors[l], len(children))):
                s, ys = children[i]
                if ys < threshold:
                    break
                if not self.hierarchy.is_dummy(s):
                    accepted_children.append(children[i])

            if len(accepted_children) == 0 and len(children) != 0:  # has children, but none accepted
                if self.strategies[l] == "top":
                    return ({children[0][0]}, True) if len(children) > 0 else (set(), True)  # enforces the top type
                elif self.strategies[l] == "other":
                    return {self.hierarchy.index("/other")}, True  # for OntoNotes
                else:
                    return set(), True

            else:
                return {s for s, _ in accepted_children}, True

        def decode_instance(y: torch.Tensor) -> Set[int]:
            beam: List[Tuple[int, int, bool]] = [(0, 0, True)]  # root (level, type, explore?)
            i = 0
            while i < len(beam):
                l, t, flag = beam[i]
                if l >= self.hierarchy.num_level - 1 or (not flag):
                    break
                if self.strategies[l] == "weak":
                    children = step_weak(y, l + 1)
                    for s in children:
                        beam.append((l + 1, s, True))
                    while i < len(beam) and beam[i][0] == l:
                        i += 1  # skip all child of this level
                else:  # strict decoding on the hierarchy
                    children, flag = step(y, l, t)
                    for s in children:
                        beam.append((l + 1, s, flag))
                    i += 1

            return set(t for l, t, _ in beam if l != 0)

        return [
            decode_instance(predictions[i, :])
            for i in range(predictions.size(0))
        ]
