import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .so3 import SO3_Embedding
from .transformer_block import *
from .layer_norm import get_normalization_layer
from .so3 import SO3_LinearV2

class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding module.
    Creates position encodings with extended frequency range
    for better temporal signal representation. Copied from original MPFlow.
    """
    def __init__(self, dim, max_period=10.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t):
        """
        Convert time to sinusoidal embedding.
        Args:
            t: Time tensor [batch_size, 1] or similar leading dimensions.
        Returns:
            embedding: Time embedding [batch_size, dim]
        """
        device = t.device
        half_dim = self.dim // 2

        # Ensure t is atleast 2D for broadcasting: [N, 1]
        if t.ndim == 1:
            t = t.unsqueeze(-1)

        frequencies = torch.exp(
            -np.log(self.max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=device) / half_dim
        )

        args = t * frequencies
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1, 0, 0)) # Pad last dimension

        return embedding


class EquivariantMPFlow(nn.Module):
    """
    Equivariant Neural network for conditional flow matching based on EquiformerV2 blocks.

    Operates on SO(3) equivariant features and incorporates time conditioning.
    Uses TransBlockV2 blocks from EquiformerV2 for core processing.
    """
    def __init__(
        self,
        ### TransBlock ###
        sphere_channels,
        attn_hidden_channels,
        num_heads,
        attn_alpha_channels, 
        attn_value_channels,
        ffn_hidden_channels,
        output_channels, 

        lmax_list,
        mmax_list,
        
        SO3_rotation,
        mappingReduced,
        SO3_grid,

        max_num_elements,
        edge_channels_list,
        use_atom_edge_embedding=True,
        use_m_share_rad=False,

        attn_activation='silu',
        use_s2_act_attn=False, 
        use_attn_renorm=True,
        ffn_activation='silu',
        use_gate_act=False, 
        use_grid_mlp=False,
        use_sep_s2_act=True,

        norm_type='rms_norm_sh',

        alpha_drop=0.0,
        drop_path_rate=0.0, 
        proj_drop=0.0, 

        ### GNN ###
        num_layers=1,
        
        ### MPFlow ###
        time_embed_dim=128,
    ):
        super(EquivariantMPFlow, self).__init__()
        
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            block = TransBlockV2(
                sphere_channels,
                attn_hidden_channels,
                num_heads,
                attn_alpha_channels,
                attn_value_channels,
                ffn_hidden_channels,
                sphere_channels, 
                lmax_list,
                mmax_list,
                SO3_rotation,
                mappingReduced,
                SO3_grid,
                max_num_elements,
                edge_channels_list,
                use_atom_edge_embedding,
                use_m_share_rad,
                attn_activation,
                use_s2_act_attn,
                use_attn_renorm,
                ffn_activation,
                use_gate_act,
                use_grid_mlp,
                use_sep_s2_act,
                norm_type,
                alpha_drop, 
                drop_path_rate,
                proj_drop
            )
            self.blocks.append(block)
        
        # Time embedding network
        self.time_embed_dim = time_embed_dim
        self.sinusoidal_time_embedding = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_ffn = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, sphere_channels * 2)
        )
        
        max_lmax = max(lmax_list)
        
        self.norm = get_normalization_layer(norm_type, lmax=max_lmax, num_channels=sphere_channels)
        
    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x, t, atomic_numbers, edge_distance, edge_index, batch):
        """
        Forward pass for the equivariant flow model using TransBlockV2 blocks.

        Args:
            x (torch.Tensor): SO3_Embedding object.
            t (torch.Tensor): Time tensor of shape [num_nodes, 1].
            atomic_numbers (torch.Tensor): Atomic numbers of shape [num_nodes].
            edge_distance (torch.Tensor): Edge distance of shape [num_nodes, num_neighbors].
            edge_index (torch.Tensor): Edge index of shape [2, num_edges].
            batch (torch.Tensor): Batch tensor of shape [num_nodes].
        Returns:
            torch.Tensor: Predicted velocity field dv/dt, same type as x.
        """
        
        # Time Conditioning
        t = self.sinusoidal_time_embedding(t)
        t = self.time_ffn(t.unsqueeze(1)) # [num_nodes, 1, sphere_channels * 2]
        scale, shift = torch.chunk(t, 2, dim=-1)
        x.embedding = x.embedding * (1 + scale)
        x.embedding[:, 0:1, :] = x.embedding[:, 0:1, :] + shift
        
        for block in self.blocks:
            x = block(x, 
                atomic_numbers,
                edge_distance,
                edge_index,
                batch=batch
            )
        
        x.embedding = self.norm(x.embedding)

        return x