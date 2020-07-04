from typing import *
from functools import reduce
import torch
import numpy as np
from bisect import bisect_left, bisect_right

from allennlp.data import Instance, DatasetReader
from allennlp.data.fields import MetadataField
from hiertype.contextualizers import Contextualizer, get_contextualizer
from hiertype.data import Hierarchy, StringNdArrayBerkeleyDBStorage
from hiertype.fields import TensorField, IntField


class CachedMentionReader(DatasetReader):

    def __init__(self,
                 hierarchy: Hierarchy,
                 delimiter: str = "/",
                 model: str = "elmo-original"
                 ):
        super(CachedMentionReader, self).__init__(lazy=True)
        self.hierarchy = hierarchy
        self.delimiter = delimiter
        self.contextualizer: Contextualizer = get_contextualizer(model, device="cpu", tokenizer_only=True)

    def _read(self, file_path: str) -> Iterable[Instance]:

        dump_path, file_path = file_path.split(':')
        with StringNdArrayBerkeleyDBStorage.open(dump_path, mode='r') as dump, \
            open(file_path, mode='r') as lines:

            for i, line in enumerate(lines):

                sentence_str, span_str, type_list_str = line.strip().split('\t')
                token_left_str, token_right_str = span_str.split(':')
                token_left, token_right = int(token_left_str), int(token_right_str)
                token_list = sentence_str.split(' ')
                span_text = token_list[token_left:token_right]
                subtoken_list, mapping = self.contextualizer.tokenize_with_mapping(token_list)

                subtoken_left = bisect_left(mapping, token_left)
                subtoken_right = bisect_right(mapping, token_right - 1)
                if subtoken_right == subtoken_left:
                    subtoken_right += 1

                sentence_repr_np: np.ndarray = dump[str(i)]
                sentence_repr = torch.from_numpy(sentence_repr_np).permute(1, 0, 2)  # R[Length, Layer, Emb]
                sentence_len = sentence_repr.size(0)
                sentence_repr = sentence_repr.reshape(sentence_len, -1)  # R[Length, Emb]

                assert subtoken_right <= sentence_repr.size(0)

                span_repr = sentence_repr[subtoken_left:subtoken_right, :]  # R[SpanLength, Emb]

                x_fields = {
                    "id": MetadataField(metadata=i),
                    "sentence_text": MetadataField(metadata=token_list),
                    "span_text": MetadataField(metadata=span_text),
                    "span_left": IntField(value=subtoken_left),
                    "span_right": IntField(value=subtoken_right),
                    "sentence": TensorField(tensor=sentence_repr, pad_dim=0),
                    "sentence_length": IntField(value=sentence_len),
                    "span": TensorField(tensor=span_repr, pad_dim=0),
                    "span_length": IntField(value=subtoken_right - subtoken_left)
                }

                all_types: List[int] = sorted(list(reduce(
                    lambda u, v: u.union(v),
                    (set(self.hierarchy.index_of_nodes_on_path(t)) for t in type_list_str.split(' '))
                )))

                y_fields = {
                    "labels": MetadataField(metadata=all_types)
                }

                yield Instance({**x_fields, **y_fields})

    def text_to_instance(self, *inputs) -> Instance:
        raise NotImplementedError

    def debug(self, file_path: str):
        for instance in self._read(file_path):
            print(f"{' '.join(instance['span_text'].metadata)}\t{' '.join(self.hierarchy.type_str(t) for t in instance['labels'].metadata)}")

