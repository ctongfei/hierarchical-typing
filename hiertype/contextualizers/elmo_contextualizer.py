from typing import *

import torch

from hiertype.contextualizers import Contextualizer
from allennlp.commands.elmo import ElmoEmbedder
from allennlp.modules.elmo import _ElmoBiLm


class ELMoContextualizer(Contextualizer):

    def __init__(self, elmo: ElmoEmbedder, device: str):
        self.elmo = elmo
        self.device = device

    @classmethod
    def from_model(cls,
                   elmo_weights_path: str,
                   elmo_options_path: str,
                   device: str,
                   tokenizer_only: str = False
                   ):
        if device == "cpu":
            device_id = -1
        else:
            device_id = int(device[5:])  # "cuda:1"

        if tokenizer_only:
            return cls(elmo=None, device=device)

        elmo = ElmoEmbedder(
            options_file=elmo_options_path,
            weight_file=elmo_weights_path,
            cuda_device=device_id
        )
        return cls(elmo=elmo, device=device)

    def tokenize_with_mapping(self, sentence: List[str]) -> Tuple[List[str], List[int]]:
        # Doesn't do anything -- retain original tokenization
        n = len(sentence)
        return sentence, [i for i in range(n)]

    def encode(self,
               sentences: List[List[str]],
               frozen: bool = True
               ) -> torch.Tensor:

        with torch.no_grad() if frozen else torch.enable_grad():
            embs, _ = self.elmo.batch_to_embeddings(sentences)  # R[Batch, Layer, Word, Emb]
            return embs
