"""Per-axis scale conditioning."""
import torch
import torch.nn as nn
from .wan_dit import sinusoidal_embedding_1d

class WanScale3DEmbeddingModel(torch.nn.Module):
    """Project a per-axis scale vector ``(B, num_axes)`` into a ``(B, dim*6)`` modulation
    vector for ``t_mod``.
    """
    def __init__(self, freq_dim=256, dim=1536, num_axes=3):
        super().__init__()
        self.freq_dim = freq_dim
        self.num_axes = num_axes
        self.linear = nn.Sequential(
            nn.Linear(freq_dim * num_axes, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )

    def forward(self, scale_cond):
        assert scale_cond.ndim == 2 and scale_cond.shape[1] == self.num_axes
        emb = torch.cat([sinusoidal_embedding_1d(self.freq_dim, scale_cond[:, i]) for i in range(self.num_axes)], dim=1)
        return self.linear(emb)

