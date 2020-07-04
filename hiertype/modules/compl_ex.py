from typing import *
import torch


class ComplEx(torch.nn.Module):

    def __init__(self, dim: int):
        super(ComplEx, self).__init__()
        self.dim = dim
        self.rel_emb = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        torch.nn.init.normal_(self.rel_emb, mean=0.0, std=0.01)

    @staticmethod
    def real_as_complex(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        re, im = x.chunk(chunks=2, dim=-1)
        return re, im

    def forward(self,
                s: torch.Tensor,  # R[Batch, Emb]
                t: torch.Tensor  # R[Batch, Emb]
                ) -> torch.Tensor:  # R[Batch]
        s_re, s_im = ComplEx.real_as_complex(s)
        t_re, t_im = ComplEx.real_as_complex(t)
        r_re, r_im = ComplEx.real_as_complex(self.rel_emb)

        y = s_re * t_re * r_re + s_im * t_im * r_re + s_re * t_im * r_im - s_im * t_re * r_im
        return torch.sum(y, dim=-1)
