from typing import *
import argparse
import logging
import json
import sys
import torch
import tqdm
from allennlp.data.iterators import BasicIterator
from hiertype.data import Hierarchy, CachedMentionReader
from hiertype.models import HierarchicalTyper
from hiertype.modules import MentionFeatureExtractor, TypeScorer
from hiertype.decoders import BeamDecoder
from fire import Fire


def main(*,
         model: str,
         model_file: str = "best.th",
         test: str,
         out: str,
         max_branching_factors: List[int],
         delta: List[float],
         strategies: List[str],
         other_delta: float = 0.0,
         seed: int = 0xDEADBEEF,
         batch_size: int = 256,
         gpuid: int = 0
         ):
    TEST_ARGS = argparse.Namespace(**locals().copy())
    ARGS = argparse.Namespace(**json.load(open(f"{TEST_ARGS.model}/args.json", mode='r')))

    for key, val in ARGS.__dict__.items():
        print(f"ARG {key}: {val}", file=sys.stderr)
    for key, val in TEST_ARGS.__dict__.items():
        print(f"TEST_ARG {key}: {val}", file=sys.stderr)

    torch.cuda.set_device(gpuid)
    torch.manual_seed(seed)

    if TEST_ARGS.max_branching_factors is None:
        TEST_ARGS.max_branching_factors = ARGS.max_branching_factors
    if TEST_ARGS.delta is None:
        TEST_ARGS.delta = ARGS.delta
    if TEST_ARGS.strategies is None:
        TEST_ARGS.strategies = ARGS.strategies

    hierarchy: Hierarchy = Hierarchy.from_tree_file(filename=ARGS.ontology, with_other=ARGS.with_other)

    model = HierarchicalTyper(
        hierarchy=hierarchy,
        input_dim=ARGS.input_dim,
        type_dim=ARGS.type_dim,
        bottleneck_dim=ARGS.bottleneck_dim,
        mention_pooling=ARGS.mention_pooling,
        with_context=True,
        dropout_rate=ARGS.dropout_rate,
        emb_dropout_rate=ARGS.emb_dropout_rate,
        margins_per_level=ARGS.margins,
        num_negative_samples=ARGS.num_negative_samples,
        threshold_ratio=ARGS.threshold_ratio,
        lift_other=ARGS.lift_other,
        relation_constraint_coef=ARGS.relation_constraint_coef,
        compute_metric_when_training=True,
        decoder=BeamDecoder(
            hierarchy=hierarchy,
            strategies=TEST_ARGS.strategies,
            max_branching_factors=TEST_ARGS.max_branching_factors,
            delta=TEST_ARGS.delta,
            top_other_delta=TEST_ARGS.other_delta
        )
    )

    model_state = torch.load(f"{ARGS.out}/{TEST_ARGS.model_file}", map_location=lambda storage, loc: storage)
    model.load_state_dict(model_state)
    model.cuda()
    model.eval()

    model.metric.set_serialization_dir(TEST_ARGS.out)
    print("Model loaded.", file=sys.stderr)

    test_reader = CachedMentionReader(hierarchy=hierarchy, model=ARGS.contextualizer)
    iterator = BasicIterator(batch_size=TEST_ARGS.batch_size)

    with torch.no_grad():
        for batch in tqdm.tqdm(iterator(instances=test_reader.read(TEST_ARGS.test), num_epochs=1, shuffle=False)):
            for k, v in batch.items():
                if hasattr(v, 'cuda'):
                    batch[k] = v.cuda()
            model(**batch)

    for m, v in model.metric.get_metric(reset=False).items():
        print(f"METRIC {m}: {v}")


Fire(main)
