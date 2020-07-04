from typing import *
from abc import abstractmethod
import torch


class Contextualizer:
    """
    Wraps around any contextualizer with optional subword tokenization.
    This abstracts over the following 3 cases:
      - GloVe et al. (no contextualization);
      - ELMo et al. (character-based);
      - BERT et al. (subword-based).

    These are encapsulated in the following 3 steps, as represented
    by the 3 abstract methods.
    This abstraction does not support methods requiring a language input,
    i.e. XLM (but it can process XLM-Roberta).
    """

    @abstractmethod
    def tokenize_with_mapping(self,
                              sentence: List[str]
                              ) -> Tuple[Union[List[int], List[str]], List[int]]:
        """
        Given a sentence tokenized into words,
        tokenizes it into subword units,
        optionally index (ELMo does not do this) these subword units into IDs,
        and returns a mapping of the tokenized symbols and the original token indices.
        :param sentence: List of original tokens
        :param lang: Language ID
        :return: (subword token indices, index mapping)
        """
        raise NotImplementedError

    @abstractmethod
    def encode(self,
               sentences: List[Union[List[int], List[str]]],
               frozen: bool = True
               ) -> torch.Tensor:  # R[Batch, Layer, Word, Emb]
        """
        Encodes these sentences, with their optional language IDs
        :param sentences:
        :param frozen: Whether the encoder is frozen
        :return:
        """
        raise NotImplementedError
