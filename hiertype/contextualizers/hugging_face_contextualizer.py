from typing import *
from abc import abstractmethod
import torch
import transformers

from hiertype.contextualizers import Contextualizer


class HuggingFaceContextualizer(Contextualizer):
    """
    Wraps around any contextualizer in the HuggingFace Transformers package.
    """

    def __init__(self,
                 hf_tokenizer: transformers.PreTrainedTokenizer,
                 hf_model: transformers.PreTrainedModel,
                 device: str
                 ):
        self.hf_tokenizer = hf_tokenizer
        self.hf_model = hf_model
        self.device = device

    @abstractmethod
    def postprocess_mapping(self,
                            mapping: List[int]
                            ) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def select_output(self, output: Any) -> torch.Tensor:
        raise NotImplementedError

    def tokenize_with_mapping(self,
                              sentence: List[str]
                              ) -> Tuple[Union[List[int]], List[int]]:
        tokens = []
        mapping = []
        for i, t in enumerate(sentence):
            for wp in self.hf_tokenizer.tokenize(t):
                tokens.append(wp)
                mapping.append(i)

        token_indices = self.hf_tokenizer.convert_tokens_to_ids(tokens)
        token_indices_with_special_symbols = self.hf_tokenizer.build_inputs_with_special_tokens(token_indices)
        mapping_with_special_symbols = self.postprocess_mapping(mapping)

        return token_indices_with_special_symbols, mapping_with_special_symbols

    def encode(self,
               indexed_sentences: List[List[int]],
               frozen: bool = True,
               ) -> torch.Tensor:  # R[Batch, Layer, Length, Emb]

        bsz = len(indexed_sentences)
        lengths = [len(s) for s in indexed_sentences]
        indices_tensor = torch.zeros(bsz, max(lengths), dtype=torch.int64)
        input_mask = torch.zeros(bsz, max(lengths), dtype=torch.int64)

        for i in range(bsz):
            for j in range(lengths[i]):
                indices_tensor[i, j] = indexed_sentences[i][j]
                input_mask[i, j] = 1

        indices_tensor = indices_tensor.to(device=self.device)
        input_mask = input_mask.to(device=self.device)

        if frozen:
            self.hf_model.eval()
        else:
            self.hf_model.train()

        with torch.no_grad() if frozen else torch.enable_grad():
            model_output = self.hf_model(
                input_ids=indices_tensor,
                attention_mask=input_mask
            )
            embs = self.select_output(model_output)
            return embs


class BERTContextualizer(HuggingFaceContextualizer):

    def __init__(self,
                 hf_tokenizer: transformers.BertTokenizer,
                 hf_model: transformers.BertModel,
                 device: str
                 ):
        super(BERTContextualizer, self).__init__(hf_tokenizer, hf_model, device)

    @classmethod
    def from_model(cls, model_name: str, device: str, tokenizer_only: bool = False):
        hf_tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
        hf_model = None if tokenizer_only \
            else transformers.BertModel.from_pretrained(model_name, output_hidden_states=True)
        if not tokenizer_only and device != "cpu":
            hf_model.cuda(device=device)
        return cls(
            hf_tokenizer=hf_tokenizer,
            hf_model=hf_model,
            device=device
        )

    def postprocess_mapping(self, mapping: List[int]) -> List[int]:
        # account for [CLS] and [SEP]
        return [-1] + mapping + [max(mapping) + 1]

    def select_output(self, output: Any) -> torch.Tensor:
        encoded = output[2]  # List_Layer[R[Batch, Word, Emb]]
        stacked = torch.stack(encoded, dim=1)  # R[Batch, Layer, Word, Emb]
        return stacked


class XLMRobertaContextualizer(HuggingFaceContextualizer):

    def __init__(self,
                 hf_tokenizer: transformers.XLMRobertaTokenizer,
                 hf_model: transformers.XLMRobertaModel,
                 device: str
                 ):
        super(XLMRobertaContextualizer, self).__init__(hf_tokenizer, hf_model, device)

    @classmethod
    def from_model(cls, model_name: str, device: str, tokenizer_only: bool = False):
        hf_tokenizer = transformers.XLMRobertaTokenizer.from_pretrained(model_name)
        hf_model = None if tokenizer_only \
            else transformers.XLMRobertaModel.from_pretrained(model_name, output_hidden_states=True)
        if not tokenizer_only and device != "cpu":
            hf_model.cuda(device=device)
        return cls(
            hf_tokenizer=hf_tokenizer,
            hf_model=hf_model,
            device=device
        )

    def postprocess_mapping(self, mapping: List[int]) -> List[int]:
        # account for <s> and </s>
        return [-1] + mapping + [max(mapping) + 1]

    def select_output(self, output: Any) -> torch.Tensor:
        encoded = output[2]  # List_Layer[R[Batch, Word, Emb]]
        stacked = torch.stack(encoded, dim=1)  # R[Batch, Layer, Word, Emb]
        return stacked
