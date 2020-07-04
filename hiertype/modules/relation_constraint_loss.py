from typing import *
import torch
import torch.nn.functional as F

from hiertype.modules import ComplEx


class RelationConstraintLoss(torch.nn.Module):

    def __init__(self,
                 type_embeddings: torch.nn.Embedding,
                 scorer: torch.nn.Module,
                 ):
        super(RelationConstraintLoss, self).__init__()
        self.scorer = scorer
        self.type_embeddings = type_embeddings

    def forward(self,
                subtype_ids: torch.Tensor,  # Z_Type[Batch, PosCand]
                pos_supertype_ids: torch.Tensor,  # Z_Type[Batch, PosCand]
                neg_supertype_ids: torch.Tensor,  # Z_Type[Batch, PosCand, NegCand]
                negative_samples_coef: float = 1.0
                ):

        batch_size = subtype_ids.size(0)
        zero = torch.tensor(0, dtype=torch.int64, device=subtype_ids.device)

        subtype_mask = subtype_ids != -1  # B[Batch, PosCand]
        pos_mask = pos_supertype_ids != -1  # B[Batch, PosCand]
        pos_mask_r = pos_mask.float()  # R[Batch, PosCand]
        neg_mask = neg_supertype_ids != -1  # B[Batch, PosCand, NegCand]
        neg_mask_r = neg_mask.float()  # R[Batch, PosCand, NegCand]

        subtype_ids = torch.where(subtype_mask, subtype_ids, zero)
        pos_supertype_ids = torch.where(pos_mask, pos_supertype_ids, zero)
        neg_supertype_ids = torch.where(neg_mask, neg_supertype_ids, zero)

        subtype_embs = self.type_embeddings(subtype_ids)  # R[Batch, Emb]
        pos_supertype_embs = self.type_embeddings(pos_supertype_ids)  # R[Batch, PosCand, Emb]
        neg_supertype_embs = self.type_embeddings(neg_supertype_ids)  # R[Batch, PosCand, NegCand, Emb]

        pos_scores = self.scorer(
            subtype_embs,
            pos_supertype_embs
        )  # R[Batch, PosCand]

        neg_scores = self.scorer(
            subtype_embs.unsqueeze(dim=2).expand_as(neg_supertype_embs),
            neg_supertype_embs
        )  # R[Batch, PosCand, NegCand]

        pos_diff = F.relu(-pos_scores + 1.0)
        pos_loss = (pos_diff * pos_mask_r).sum() / pos_mask_r.sum()

        neg_diff = F.relu(neg_scores + 1.0)
        neg_loss = (neg_diff * neg_mask_r).sum() / neg_mask_r.sum() * negative_samples_coef

        return pos_loss + neg_loss
