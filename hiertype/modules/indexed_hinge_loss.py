from typing import *

import torch
import torch.nn.functional as F


class IndexedHingeLoss(torch.nn.Module):

    def __init__(self,
                 margins: torch.Tensor,  # R[Level]
                 ):
        super(IndexedHingeLoss, self).__init__()
        self.margins = torch.nn.Parameter(margins)
        self.margins.requires_grad = False  # freeze!

    def forward(self,
                scores: torch.Tensor,  # R[Batch, Type]
                pos_type_ids: torch.Tensor,  # Z_Type[Batch, PosCand]
                neg_type_ids: torch.Tensor,  # Z_Type[Batch, PosCand, NegCand]
                levels: torch.Tensor,  # Z_Level[Batch, PosCand]
                margin_ratio: float
                ):

        batch_size = scores.size(0)
        max_pos_type_size = pos_type_ids.size(1)
        max_neg_type_size = neg_type_ids.size(2)
        device = scores.device
        batch_range = torch.arange(0, batch_size, dtype=torch.int64, device=device)  # Z_Batch[Batch]
        zero = torch.tensor(0, dtype=torch.int64, device=device)

        neg_mask = (neg_type_ids != -1)  # B[Batch, PosCand, NegCand]
        neg_mask_r = neg_mask.float()  # R[Batch, PosCand, NegCand]
        pos_mask = pos_type_ids != -1

        pos_type_ids = torch.where(pos_mask, pos_type_ids, zero)
        neg_type_ids = torch.where(neg_mask, neg_type_ids, zero)

        pos_scores = scores[
            batch_range.unsqueeze(dim=1).expand(batch_size, max_pos_type_size),
            pos_type_ids
        ]  # R[Batch, PosCand]

        neg_scores = scores[
            batch_range.unsqueeze(dim=1).unsqueeze(dim=2).expand(batch_size, max_pos_type_size, max_neg_type_size),
            neg_type_ids
        ]  # R[Batch, PosCand, NegCand]

        level_margins = self.margins[levels] * margin_ratio  # R[Batch, PosCand]

        diff = F.relu(
            level_margins.unsqueeze(dim=2).expand_as(neg_scores)
            - pos_scores.unsqueeze(dim=2).expand_as(neg_scores)
            + neg_scores
        )

        return (diff * neg_mask_r).sum() / neg_mask_r.sum()
