from typing import *

from functools import reduce
import collections
from allennlp.training.metrics import Metric

from hiertype.data import Hierarchy
from hiertype.metrics import SetMetric


class HierarchicalMetric(Metric):

    def __init__(self, hierarchy: Hierarchy):
        self.hierarchy = hierarchy
        self.num_level = hierarchy.num_level
        self.level_sets = [
            set(self.hierarchy.level_range(l))
            for l in range(self.num_level)
        ]

        self.metrics_by_level = [SetMetric() for l in range(self.num_level)]
        self.overall_metric = SetMetric()

        self.serialization_dir: Optional[str] = None
        self.output: Optional[IO] = None

    def set_serialization_dir(self, serialization_dir: str):
        self.serialization_dir = serialization_dir

    def init_output(self, partition: str, epoch: int):
        if self.output is None:
            self.output = open(f"{self.serialization_dir}/{partition}-{epoch}.res", mode='w')

    def __call__(self, pred: List[Set[int]], gold: List[Set[int]]):

        # Remove root node 0 and OTHER
        pred: List[Set[int]] = [
            {t for t in ts if not self.hierarchy.is_dummy(t)}
            for ts in pred
        ]
        gold: List[Set[int]] = [
            {t for t in ts if not self.hierarchy.is_dummy(t)}
            for ts in gold
        ]

        batch_size = len(pred)
        for b in range(batch_size):

            # self.n += 1
            # metrics by level
            instance_pred_by_level = [{0}]
            instance_gold_by_level = [{0}]
            for l in range(1, self.num_level):
                instance_pred_l = pred[b] & self.level_sets[l]
                instance_pred_l |= instance_pred_by_level[l - 1] - \
                                   {self.hierarchy.parent(t) for t in instance_pred_l}
                # add those in the upper layer with no children
                instance_gold_l = gold[b] & self.level_sets[l]
                instance_gold_l |= instance_gold_by_level[l - 1] - \
                                   {self.hierarchy.parent(t) for t in instance_gold_l}

                pred_str = ' '.join(sorted(self.hierarchy.type_str(t) for t in instance_pred_l))
                gold_str = ' '.join(sorted(self.hierarchy.type_str(t) for t in instance_gold_l))
                print(f"[{l}]\t{pred_str}\t|\t{gold_str}", file=self.output)

                self.metrics_by_level[l]([instance_pred_l], [instance_gold_l])

                instance_pred_by_level.append(instance_pred_l)
                instance_gold_by_level.append(instance_gold_l)

            pred_str = ' '.join(sorted(self.hierarchy.type_str(t) for t in pred[b]))
            gold_str = ' '.join(sorted(self.hierarchy.type_str(t) for t in gold[b]))
            print(f"[+]\t{pred_str}\t|\t{gold_str}", file=self.output)

            self.overall_metric([pred[b]], [gold[b]])

    def get_metric(self, reset: bool) -> Dict[str, float]:

        level_metrics = {
            f"L{l}_{k}": v
            for l in range(1, self.num_level)
            for k, v in self.metrics_by_level[l].get_metric(reset).items()
        }

        overall_metrics = {
            f"O_{k}": v for k, v in self.overall_metric.get_metric(reset).items()
        }
        if reset:
            self.reset()

        return collections.OrderedDict(sorted({**level_metrics, **overall_metrics}.items()))

    def reset(self) -> None:
        """
        Flushes the output file that stores prediction results.
        """
        if self.output is not None:
            self.output.close()
        self.output = None
