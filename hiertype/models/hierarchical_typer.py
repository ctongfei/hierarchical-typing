from typing import *
import torch
import numpy as np
import json
from allennlp.models import Model

from hiertype.data import Hierarchy
from hiertype.modules import MentionFeatureExtractor, TypeScorer, IndexedHingeLoss, ComplEx, RelationConstraintLoss
from hiertype.decoders import HierarchyDecoder, BeamDecoder
from hiertype.metrics import HierarchicalMetric
from hiertype.training import MyTrainer
from hiertype.util import compact2, compact3, sample_multiple_from, sample_in_range_except


class HierarchicalTyper(Model):

    def __init__(self,
                 hierarchy: Hierarchy,
                 input_dim: int,
                 type_dim: int,
                 bottleneck_dim: int,
                 mention_pooling: str,
                 with_context: bool,
                 dropout_rate: float,
                 emb_dropout_rate: float,
                 margins_per_level: List[float],
                 num_negative_samples: int,
                 threshold_ratio: float,
                 relation_constraint_coef: float,
                 lift_other: bool,
                 compute_metric_when_training: bool,
                 decoder: HierarchyDecoder
                 ):
        super(HierarchicalTyper, self).__init__(vocab=None)

        self.hierarchy = hierarchy
        self.threshold_ratio = threshold_ratio
        self.relation_constraint_coef = relation_constraint_coef
        self.num_negative_samples = num_negative_samples
        self.lift_other = lift_other

        self.mention_feature_extractor = MentionFeatureExtractor(
            hierarchy=hierarchy,
            dim=input_dim,
            dropout_rate=emb_dropout_rate,
            mention_pooling=mention_pooling,
            with_context=with_context
        )
        self.type_scorer = TypeScorer(
            type_embeddings=torch.nn.Embedding(hierarchy.size(), type_dim),
            input_dim=(input_dim * 2) if with_context else input_dim,
            type_dim=type_dim,
            bottleneck_dim=bottleneck_dim,
            dropout_rate=dropout_rate
        )
        self.loss = IndexedHingeLoss(torch.tensor([0.0] + margins_per_level, dtype=torch.float32))
        self.rel_loss = RelationConstraintLoss(
            self.type_scorer.type_embeddings,
            ComplEx(self.type_scorer.type_embeddings.embedding_dim)
        )

        self.decoder = decoder
        self.compute_metric_when_training = compute_metric_when_training
        self.metric = HierarchicalMetric(hierarchy)

        self.trainer: MyTrainer = None
        self.current_epoch = 0

    def set_trainer(self, trainer: MyTrainer):
        self.trainer = trainer

    def scores(self,
               sentence: torch.Tensor,
               sentence_length: torch.Tensor,
               span: torch.Tensor,
               span_length: torch.Tensor,
               span_left: torch.Tensor,  # Z[Batch]
               span_right: torch.Tensor,  # Z[Batch]
               **kwargs
               ) -> torch.Tensor:  # R[Batch, Type]

        mention_features = self.mention_feature_extractor(
            sentence, sentence_length, span, span_length, span_left, span_right
        )  # R[Batch, Emb]
        scores = self.type_scorer(mention_features)  # R[Batch, Type]
        return scores

    def forward(self,
                id: List[int],
                span_text: List[List[str]],
                sentence_text: List[List[str]],
                sentence: torch.Tensor,
                sentence_length: torch.Tensor,
                span: torch.Tensor,
                span_length: torch.Tensor,
                span_left: torch.Tensor,  # Z[Batch]
                span_right: torch.Tensor,  # Z[Batch]
                labels: List[List[int]]
                ) -> Dict[str, torch.Tensor]:

        device = sentence.device

        scores = self.scores(sentence, sentence_length, span, span_length, span_left, span_right)

        pos_type_ids = torch.from_numpy(self.get_pos_indices(labels, lift_other=self.lift_other)).to(device=device)
        thr_type_ids = torch.from_numpy(self.get_parent_indices(labels)).to(device=device)
        neg_type_ids = torch.from_numpy(self.get_neg_sibling_indices(labels)).to(device=device)
        neg_parent_type_ids = torch.from_numpy(self.get_parent_sibling_indices(labels)).to(device=device)
        levels = torch.from_numpy(self.levels(labels)).to(device=device)

        loss_above = self.loss(scores, pos_type_ids, thr_type_ids.unsqueeze(dim=2), levels, self.threshold_ratio)
        loss_below = self.loss(scores, thr_type_ids, neg_type_ids, levels, 1.0 - self.threshold_ratio)
        loss_both = self.loss(scores, pos_type_ids, neg_type_ids, levels, 1.0)

        rel_sibling_loss = self.rel_loss(pos_type_ids, thr_type_ids, neg_type_ids)  # siblings are not parent
        rel_parent_loss = self.rel_loss(pos_type_ids, thr_type_ids, neg_parent_type_ids)  # siblings of parent are not parent
        all_rel_loss = rel_sibling_loss + rel_parent_loss

        return_dict = {
            "scores": scores,
            "loss_above": loss_above,
            "loss_below": loss_below,
            "loss_both": loss_both,
            "loss_rel": all_rel_loss,
            "loss": loss_above + loss_below + loss_both + all_rel_loss * self.relation_constraint_coef
        }

        if self.training and not self.compute_metric_when_training:
            return return_dict

        predicted_types = self.decoder.decode(scores)
        self.current_epoch = 0 if not self.trainer else self.trainer.current_epoch
        if self.training:
            self.metric.init_output("train", self.current_epoch)
        else:
            self.metric.init_output("dev", self.current_epoch)
        self.metric(predicted_types, labels)

        return return_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self.metric.get_metric(reset)

    def levels(self, labels: List[List[int]]) -> np.ndarray:
        _, l = compact2(
            [
                [self.hierarchy.level(x) for x in xs if x != 0]
                for xs in labels
            ],
            pad=0
        )
        return l

    def get_pos_indices(self, labels: List[List[int]], lift_other: bool = False) -> np.ndarray:
        def lift(x):
            if self.hierarchy.is_dummy(x):
                return self.hierarchy.parent(x)
            else:
                return x

        _, indices = compact2(
            [
                [lift(x) if lift_other else x for x in xs if x != 0]  # remove root, which has id 0
                for xs in labels
            ],
            pad=-1
        )
        return indices  # Z_Type[Batch, PosCand]

    def get_parent_indices(self, labels: List[List[int]]) -> np.ndarray:
        _, indices = compact2(
            [
                [self.hierarchy.parent(x) for x in xs if x != 0]  # remove root, which has id 0
                for xs in labels
            ],
            pad=-1
        )
        return indices

    def get_neg_sibling_indices(self, labels: List[List[int]]) -> np.ndarray:

        label_sets: List[Set[int]] = [set(xs) for xs in labels]

        # use all negative siblings of this positive root if num_negative_samples <= 0
        def choose_except(x: int, excluded: Set[int]) -> List[int]:
            if self.num_negative_samples <= 0:
                return list(self.hierarchy.sibling(x).difference(excluded))
            else:
                r = self.hierarchy.level_range(self.hierarchy.level(x))
                return sample_in_range_except(r.start, r.stop, self.num_negative_samples, excluded)

        l = [
            [
                choose_except(x, label_sets[i])
                for x in xs if x != 0
            ]
            for i, xs in enumerate(labels)
        ]
        _, _, indices = compact3(l, pad=-1)
        return indices

    def get_parent_sibling_indices(self, labels: List[List[int]]) -> np.ndarray:

        label_sets: List[Set[int]] = [set(xs) for xs in labels]

        # use all negative siblings of this positive root if num_negative_samples <= 0
        def choose_except(x: int, excluded: Set[int]) -> List[int]:
            if self.num_negative_samples <= 0:
                return list(self.hierarchy.sibling(self.hierarchy.parent(x)).difference(excluded))
            else:
                r = self.hierarchy.level_range(self.hierarchy.level(x) - 1)
                return sample_in_range_except(r.start, r.stop, self.num_negative_samples, excluded)

        l = [
            [
                choose_except(x, label_sets[i])
                for x in xs if x != 0
            ]
            for i, xs in enumerate(labels)
        ]

        _, _, indices = compact3(l, pad=-1)
        return indices

    @classmethod
    def from_args(cls, args_path: str):
        args = json.load(open(args_path))
        hierarchy = Hierarchy.from_tree_file(args["ontology"], with_other=args["with_other"])
        return cls(
            hierarchy=hierarchy,
            input_dim=args["input_dim"],
            type_dim=args["type_dim"],
            bottleneck_dim=args["bottleneck_dim"],
            mention_pooling=args["mention_pooling"],
            with_context=True,
            dropout_rate=args["dropout_rate"],
            emb_dropout_rate=args["emb_dropout_rate"],
            margins_per_level=args["margins"],
            num_negative_samples=args["num_negative_samples"],
            threshold_ratio=args["threshold_ratio"],
            relation_constraint_coef=args["relation_constraint_coef"],
            lift_other=args["lift_other"],
            compute_metric_when_training=True,
            decoder=BeamDecoder(
                hierarchy=hierarchy,
                strategies=args["strategies"],
                max_branching_factors=args["max_branching_factors"],
                delta=args["delta"]
            )
        )

    @classmethod
    def from_model_path(cls, model_path: str, device: str = None):
        model = cls.from_args(f"{model_path}/args.json")
        model.load_state_dict(torch.load(f"{model_path}/best.th"))
        if device is not None:
            model.cuda(device)
        return model
