from typing import *
import torch


class TypeScorer(torch.nn.Module):

    def __init__(self,
                 type_embeddings: torch.nn.Embedding,
                 input_dim: int,
                 type_dim: int,
                 bottleneck_dim: int,
                 dropout_rate: float
                 ):
        super(TypeScorer, self).__init__()
        self.type_embeddings = type_embeddings
        self.ffnn = torch.nn.Sequential(
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(input_dim, input_dim // 2),
            torch.nn.Tanh(),
            torch.nn.Linear(input_dim // 2, input_dim // 2),
            torch.nn.Tanh(),
            torch.nn.Linear(input_dim // 2, type_dim),
            torch.nn.Tanh()
        )
        self.linear = torch.nn.Linear(
            in_features=type_embeddings.embedding_dim,
            out_features=type_embeddings.num_embeddings,
            bias=True
        )
        self.linear.weight = type_embeddings.weight  # Put the embeddings into the last layer
        self.bottleneck_dim = bottleneck_dim

        if self.bottleneck_dim > 0:
            self.bottleneck_weight = torch.nn.Parameter(torch.tensor(0.1))
            self.bottleneck = torch.nn.Sequential(
                torch.nn.Linear(type_embeddings.embedding_dim, bottleneck_dim),
                torch.nn.Linear(bottleneck_dim, type_embeddings.num_embeddings)
            )

    def forward(self,
                features: torch.Tensor
                ) -> torch.Tensor:

        mapped_mentions = self.ffnn(features)  # R[Batch, Emb]
        scores = self.linear(mapped_mentions)  # R[Batch, Type]

        if self.bottleneck_dim > 0:
            bottleneck_scores = self.bottleneck(mapped_mentions)  # R[Batch, Type]
            scores = scores + self.bottleneck_weight * bottleneck_scores

        return scores

