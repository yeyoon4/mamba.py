import torch
from mambapy.mamba import Mamba, MambaConfig

config = MambaConfig(d_model=2560, n_layers=2)
model = Mamba(config)

B, L, D = 1, 32, 2560
x = torch.randn(B, L, D)
y = model(x)

assert y.shape == x.shape