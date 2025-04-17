import torch

from nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20
from e3nn import o3
from types import SimpleNamespace
import numpy as np
import random
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
seed_everything(42)

model_config = {
    "use_pbc": True, 
    "regress_forces": True,
    "otf_graph": True,  
    "max_neighbors": 20,  
    "max_radius": 12.0,  
    "max_num_elements": 23,  # this is what I use for the custom dataset with the highest atomic number being 22; I tried different values when inputting a random sample
    "num_layers": 2,
    "sphere_channels": 128,
    "attn_hidden_channels": 64,
    "num_heads": 8,
    "attn_alpha_channels": 64,
    "attn_value_channels": 16,
    "ffn_hidden_channels": 128,
    "norm_type": "layer_norm_sh",
    "lmax_list": [4],
    "mmax_list": [2],
    "grid_resolution": 18,
    "num_sphere_samples": 128,
    "edge_channels": 128,
    "use_atom_edge_embedding": True,
    "share_atom_edge_embedding": False,
    "distance_function": "gaussian",
    "num_distance_basis": 512,
    "attn_activation": "silu",
    "use_s2_act_attn": False,
    "use_attn_renorm": True,
    "ffn_activation": "silu",
    "use_gate_act": False,
    "use_grid_mlp": True,
    "use_sep_s2_act": True,
    "alpha_drop": 0.1,
    "drop_path_rate": 0.1,
    "proj_drop": 0.0,
    "weight_init": "uniform",
}
model = EquiformerV2_OC20(None, None, None, **model_config).to(device) # also tried with the standard params in the constructor, i.e., without model_config

N = 10 # num atoms

data = {
    "pos": torch.randn(N, 3),
    "pbc": torch.tensor([[True, True, True]]), 
    "atomic_numbers": torch.ones(N).long(), 
    "cell": torch.randn(1, 3, 3), 
    "natoms": torch.tensor([N]),
    "batch": torch.zeros(N).long(),
}

data = {k: v.to(device) for k, v in data.items()}

# convert the input to the right format:
data = SimpleNamespace(**data)

# SO(3)-equivariance test:
pos = data.pos
cell = data.cell
R = torch.tensor(o3.rand_matrix()).to(device) # det(R) = +1
model.eval()
with torch.no_grad():
    energy1, forces1, *_ = model(data, predict_with_mpflow=True)
    
    rotated_pos = torch.matmul(pos, R)
    rotated_cell = torch.matmul(cell, R)
    data.pos = rotated_pos
    data.cell = rotated_cell

    energy2, forces2, *_ = model(data, predict_with_mpflow=True)
    
print(energy1, energy2)

print(f"\nInvariance error:\n(energy1 - energy2).abs() = {(energy1 - energy2).abs()}")

print(f"\nEquivariance error:\n(forces1 - forces2 @ R.T).abs().max() = \
       {(forces1 - torch.matmul(forces2, R.transpose(-1, -2)).detach() ).abs().max()}")

print(f"\njust in case:\n(forces1 - forces2 @ R).abs().max() = \
       {(forces1 - torch.matmul(forces2, R).detach() ).abs().max()}")
