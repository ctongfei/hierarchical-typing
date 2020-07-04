from typing import *

import torch
from allennlp.training.metrics import Metric


class SetMetric(Metric):

    def __init__(self):
        self.n = 0
        self.strict = 0
        self.pred = 0
        self.gold = 0
        self.both = 0
        self.sum_p = 0.0
        self.sum_r = 0.0
        self.sum_f = 0.0

    def __call__(self, pred: List[Set[int]], gold: List[Set[int]]):

        batch_size = len(pred)
        for b in range(batch_size):

            self.n += 1

            pred_count = len(pred[b])
            gold_count = len(gold[b])
            both_count = len(pred[b] & gold[b])

            self.strict += 1 if pred[b] == gold[b] else 0
            self.pred += pred_count
            self.gold += gold_count
            self.both += both_count

            p = _safe_div(both_count, pred_count)
            r = _safe_div(both_count, gold_count)
            f = _f1(p, r)

            self.sum_p += p
            self.sum_r += r
            self.sum_f += f

    def get_metric(self, reset: bool) -> Dict[str, float]:
        mi_p = _safe_div(self.both, self.pred)
        mi_r = _safe_div(self.both, self.gold)
        mi_f = _f1(mi_p, mi_r)
        m = {
            "Acc": _safe_div(self.strict, self.n),
            "MaP": _safe_div(self.sum_p, self.n),
            "MaR": _safe_div(self.sum_r, self.n),
            "MaF": _safe_div(self.sum_f, self.n),
            "MiP": mi_p,
            "MiR": mi_r,
            "MiF": mi_f
        }

        if reset:
            self.reset()

        return m

    def reset(self) -> None:
        self.n = 0
        self.strict = 0
        self.pred = 0
        self.gold = 0
        self.both = 0
        self.sum_p = 0
        self.sum_r = 0
        self.sum_f = 0


def _safe_div(nominator: Union[int, float], denominator: Union[int, float]) -> float:
    return 0.0 if nominator == 0 or denominator == 0 else float(nominator) / float(denominator)


def _f1(p: float, r: float) -> float:
    return 0.0 if p == 0.0 or r == 0.0 else 2 * p * r / (p + r)
