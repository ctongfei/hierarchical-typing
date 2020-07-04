from typing import *
import torch
import torch.nn.functional as F

from hiertype.data import Hierarchy
from allennlp.nn.util import get_mask_from_sequence_lengths


class MentionFeatureExtractor(torch.nn.Module):

    def __init__(self,
                 hierarchy: Hierarchy,
                 dim: int,
                 dropout_rate: float,
                 mention_pooling: str = "max",  # max / mean / attention
                 with_context: bool = False
                 ):
        super(MentionFeatureExtractor, self).__init__()
        self.hierarchy = hierarchy
        self.mention_pooling = mention_pooling
        self.dim = dim
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.projection = torch.nn.Linear(self.dim, self.dim)
        torch.nn.init.eye_(self.projection.weight)
        torch.nn.init.zeros_(self.projection.weight)

        if self.mention_pooling == "attention":
            self.query = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float32))
            torch.nn.init.normal_(self.query, mean=0.0, std=0.02)
        self.with_context = with_context
        if self.with_context:
            self.mention_query_transform = torch.nn.Linear(self.dim, self.dim)
            torch.nn.init.normal_(self.mention_query_transform.weight, mean=0.0, std=0.02)
            torch.nn.init.zeros_(self.mention_query_transform.bias)

    def forward(self,
                sentence: torch.Tensor,  # R[Batch, Word, Emb]
                sentence_lengths: torch.Tensor,  # Z_Word[Batch]
                span: torch.Tensor,  # R[Batch, Word, Emb]
                span_lengths: torch.Tensor,  # Z_Word[Batch]
                span_left: torch.Tensor,  # Z_Word[Batch]
                span_right: torch.Tensor  # Z_Word[Batch]
                ) -> torch.Tensor:  # R[Batch, Feature]

        batch_size = sentence.size(0)
        sentence_max_len = sentence.size(1)
        emb_size = sentence.size(2)
        span_max_len = span.size(1)
        device = sentence.device
        neg_inf = torch.tensor(-10000, dtype=torch.float32, device=device)
        zero = torch.tensor(0, dtype=torch.float32, device=device)

        span = self.projection(self.dropout(span))
        sentence = self.projection(self.dropout(sentence))

        span_mask = get_mask_from_sequence_lengths(span_lengths, span_lengths.max().item()).byte()  # Z[Batch, Word]

        def attention_pool():
            span_attn_scores = torch.einsum('e,bwe->bw', self.query, span)
            masked_span_attn_scores = torch.where(span_mask, span_attn_scores, neg_inf)
            normalized_span_attn_scores = F.softmax(masked_span_attn_scores, dim=1)
            span_pooled = torch.einsum('bwe,bw->be', span, normalized_span_attn_scores)
            return span_pooled

        span_pooled = {
            "max": lambda: torch.max(torch.where(span_mask.unsqueeze(dim=2).expand_as(span), span, neg_inf), dim=1)[0],
            "mean": lambda: torch.sum(
                torch.where(span_mask.unsqueeze(dim=2).expand_as(span), span, zero), dim=1
            ) / span_lengths.unsqueeze(dim=1).expand(batch_size, emb_size),
            "attention": lambda: attention_pool()
        }[self.mention_pooling]()  # R[Batch, Emb]

        features = span_pooled

        if self.with_context:
            sentence_mask = get_mask_from_sequence_lengths(sentence_lengths, sentence_max_len).bool()  # B[B, L]

            length_range = torch.arange(0, sentence_max_len, device=device) \
                .unsqueeze(dim=0).expand(batch_size, sentence_max_len)
            span_mask = (length_range >= (span_left.unsqueeze(dim=1).expand_as(length_range))) \
                & (length_range < (span_right.unsqueeze(dim=1).expand_as(length_range)))  # B[Batch, Length]

            span_queries = self.mention_query_transform(span_pooled)
            attn_scores = torch.einsum('be,bwe->bw', span_queries, sentence)  # R[Batch, Word]
            masked_attn_scores = torch.where(sentence_mask, attn_scores, neg_inf)  # R[Batch, Word]  & ~span_mask
            normalized_attn_scores = F.softmax(masked_attn_scores, dim=1)
            context_pooled = torch.einsum('bwe,bw->be', sentence, normalized_attn_scores)  # R[Batch, Emb]

            features = torch.cat([span_pooled, context_pooled], dim=1)  # R[Batch, Emb*2]

        return features  # R[Batch, Emb]
