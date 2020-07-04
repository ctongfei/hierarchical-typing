from typing import *
import argparse
import logging
import torch
import json
import sys
import numpy as np
import random
from colors import blue
from fire import Fire

from torch.optim.optimizer import Optimizer
from torch.optim.adamw import AdamW
from allennlp.data.iterators import BasicIterator

from hiertype.training import MyTrainer
from hiertype.data import Hierarchy, CachedMentionReader
from hiertype.decoders import BeamDecoder
from hiertype.models import HierarchicalTyper


def main(*,
         ontology: str,
         train: str,
         dev: str,
         out: str,

         contextualizer: str = "elmo-original",
         input_dim: int = 3072,
         type_dim: int = 1024,
         bottleneck_dim: int = 0,
         with_other: bool = False,
         lift_other: bool = False,
         mention_pooling: str = "max",
         emb_dropout_rate: float = 0.3,
         dropout_rate: float = 0.3,

         margins: List[float] = [],
         threshold_ratio: float = 0.1,
         relation_constraint_coef: float = 0.1,
         num_negative_samples: int = 0,

         max_branching_factors: List[int] = [],
         delta: List[float] = [],
         strategies: List[str] = [],

         seed: int = 0xDEADBEEF,
         batch_size: int = 256,
         dev_batch_size: int = 256,
         num_epochs: int = 5,
         dev_metric: str = "+O_MiF",
         patience: int = 4,
         lr: float = 1e-5,
         regularizer: float = 0.1,
         gpuid: int = 0
         ):

    args = locals().copy()

    with open(f"{out}/args.json", mode='w') as args_out:
        for k, v in reversed(list(args.items())):  # seems that `locals()` stores the args in reverse order
            print(f"{blue('--' + k)} \"{v}\"", file=sys.stderr)
        print(json.dumps(args, indent=2), file=args_out)

    torch.cuda.set_device(gpuid)

    # Ensure deterministic behavior
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logging.basicConfig(level=logging.INFO)

    hierarchy: Hierarchy = Hierarchy.from_tree_file(ontology, with_other=with_other)
    print(hierarchy, file=sys.stderr)

    reader = CachedMentionReader(hierarchy, model=contextualizer)

    model = HierarchicalTyper(
        hierarchy=hierarchy,
        input_dim=input_dim,
        type_dim=type_dim,
        bottleneck_dim=bottleneck_dim,
        mention_pooling=mention_pooling,
        with_context=True,
        dropout_rate=dropout_rate,
        emb_dropout_rate=emb_dropout_rate,
        margins_per_level=margins,
        num_negative_samples=num_negative_samples,
        threshold_ratio=threshold_ratio,
        relation_constraint_coef=relation_constraint_coef,
        lift_other=lift_other,
        compute_metric_when_training=True,
        decoder=BeamDecoder(
            hierarchy=hierarchy,
            strategies=strategies,
            max_branching_factors=max_branching_factors,
            delta=delta
        )
    )
    model.cuda()

    optimizer: Optimizer = AdamW(
        params=model.parameters(),
        lr=lr,
        weight_decay=regularizer
    )

    trainer = MyTrainer(
        model=model,
        optimizer=optimizer,
        iterator=BasicIterator(batch_size=batch_size),
        validation_iterator=BasicIterator(batch_size=dev_batch_size),
        train_dataset=reader.read(train),
        validation_dataset=reader.read(dev),
        validation_metric=dev_metric,
        patience=patience,
        num_epochs=num_epochs,
        grad_norm=1.0,
        serialization_dir=out,
        num_serialized_models_to_keep=1,
        cuda_device=gpuid
    )

    model.set_trainer(trainer)
    model.metric.set_serialization_dir(trainer._serialization_dir)
    # hook into the trainer to set the metric serialization path
    trainer.train()


if __name__ == "__main__":
    Fire(main)
