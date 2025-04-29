import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .so3 import SO3_Embedding
from .transformer_block import FeedForwardNetwork
from torch.nn import ModuleList
from .layer_norm import get_normalization_layer
from .so3 import SO3_LinearV2

class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding module.
    Creates position encodings with extended frequency range
    for better temporal signal representation. Copied from original MPFlow.
    """
    def __init__(self, dim, max_period=10000.0):
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
        ### FeedForwardNetwork ###
        sphere_channels,
        hidden_channels_list, 
        output_channels,
        lmax_list,
        mmax_list,
        SO3_grid,  
        activation='scaled_silu', 
        use_gate_act=False, 
        use_grid_mlp=False, 
        use_sep_s2_act=True,
        
        ### Normalization ###
        norm_type='layer_norm_sh',

        ### MPFlow ###
        time_embed_dim=128,
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.total_coefficients = sum([(l+1)**2 for l in lmax_list]) # Recalculate based on actual list
        self.time_embed_dim = time_embed_dim
        self.num_layers = len(hidden_channels_list) + 1
        
        # Time embedding network
        self.sinusoidal_time_embedding = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_ffn = FeedForwardNetwork(
            time_embed_dim,
            time_embed_dim * 2,
            sphere_channels * 2,
            [0],
            [0],
            SO3_grid,
            activation,
            use_gate_act,
            use_grid_mlp,
            use_sep_s2_act
        )

        # Stack of TransBlockV2 blocks
        self.blocks = ModuleList()
        self.norms = ModuleList()
        self.ffn_shortcuts = ModuleList()
        
        # Define the sequence of channel dimensions
        all_channels = [sphere_channels] + hidden_channels_list + [output_channels]
        
        # Initial normalization layer
        norm = get_normalization_layer(norm_type, lmax=max(lmax_list), num_channels=all_channels[0])
        self.norms.append(norm)
        
        # Create blocks, shortcuts, and norms in a loop
        # num_layers + 1 blocks in total (input -> h1 -> ... -> hn -> output)
        for idx in range(self.num_layers):
            in_channels, out_channels = all_channels[idx], all_channels[idx+1]
            
            block = FeedForwardNetwork(
                in_channels,
                out_channels, 
                out_channels,
                lmax_list,
                mmax_list,
                SO3_grid,  
                activation,
                use_gate_act,
                use_grid_mlp,
                use_sep_s2_act
            )
            self.blocks.append(block)
            
            if in_channels != out_channels:
                ffn_shortcut = SO3_LinearV2(in_channels, out_channels, lmax=max(lmax_list))
                self.ffn_shortcuts.append(ffn_shortcut)
            else:
                self.ffn_shortcuts.append(None)
            
            norm = get_normalization_layer(norm_type, lmax=max(lmax_list), num_channels=out_channels)
            self.norms.append(norm)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x, t):
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
        # Check input shapes
        num_nodes, num_coeffs, num_channels = x.embedding.shape
        assert num_coeffs == self.total_coefficients, f"Input coefficient mismatch: {num_coeffs} vs {self.total_coefficients}"
        assert num_channels == self.sphere_channels, f"Input channel mismatch: {num_channels} vs {self.sphere_channels}"
        assert t.shape == (num_nodes, 1), f"Time tensor shape mismatch: {t.shape} vs ({num_nodes}, 1)"

        # 1. Compute time features (invariant)
        t = self.sinusoidal_time_embedding(t)
        t_embedding = SO3_Embedding(
            0,
            [0],
            self.time_embed_dim,
            device=x.device,
            dtype=x.dtype
        )
        t_embedding.set_embedding(t.unsqueeze(1))
        t_feat = self.time_ffn(t_embedding).embedding # [num_nodes, 1, sphere_channels * 2]
        scale, shift = torch.chunk(t_feat, 2, dim=-1)
        
        h = x.clone()

        # Normalize
        h_norm = self.norms[0](h.embedding)
        
        # Time conditioning (Scale and Shift)
        h_timed = h_norm * (1 + scale)
        h_timed[:, 0:1, :] = h_timed[:, 0:1, :] + shift
        h.embedding = h_timed
        
        for i, block in enumerate(self.blocks):
            x_res = h.embedding # Store residual input

            # Apply Equivariant Block
            h = block(h)
            
            # Handle FFN shortcut
            if self.ffn_shortcuts[i] is not None:
                shortcut_embedding = SO3_Embedding(
                    0,
                    h.lmax_list.copy(),
                    self.ffn_shortcuts[i].in_features,
                    device=h.device,
                    dtype=h.dtype
                )
                shortcut_embedding.set_embedding(x_res)
                shortcut_embedding.set_lmax_mmax(h.lmax_list.copy(), h.lmax_list.copy())
                shortcut_embedding = self.ffn_shortcuts[i](shortcut_embedding)
                x_res = shortcut_embedding.embedding

            # Add Residual
            h.embedding = h.embedding + x_res
            
            # # Normalize
            # h.embedding = self.norms[i+1](h.embedding)

        return h