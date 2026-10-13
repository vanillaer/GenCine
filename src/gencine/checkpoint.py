"""Strict loading of the four inference components."""
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from .models.wan_dit import WanModel
from .models.video3d_vace import VaceWanModel
from .models.scale3d import WanScale3DEmbeddingModel
from .models.wan_vae_decoder import WanVideoVAE


def load_models(weights):
    weights = Path(weights)
    config = json.loads((weights / 'model.json').read_text())
    assert config['schema_version'] == 1 and config['dtype'] == 'bfloat16'
    assert config['attention'] == 'torch_sdpa'
    models = {}
    for name, cls in [('dit', WanModel), ('vace', VaceWanModel),
                      ('scale3d', WanScale3DEmbeddingModel), ('vae_decoder', WanVideoVAE)]:
        with torch.device('meta'):
            model = cls(**config[name])
        model.load_state_dict(load_file(str(weights / f'{name}.safetensors')), strict=True, assign=True)
        assert all(p.dtype == torch.bfloat16 for p in model.parameters())
        models[name] = model.eval().requires_grad_(False)
    return models
