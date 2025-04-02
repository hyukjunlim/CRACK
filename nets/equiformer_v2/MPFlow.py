import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .so3 import SO3_Embedding, SO3_Grid, SO3_Rotation, SO3_LinearV2
from .transformer_block import FeedForwardNetwork # Local import for clarity
from torch.nn import SiLU, Linear, Dropout, ModuleList
from .layer_norm import get_normalization_layer

class MPFlow(nn.Module):
    """
    Neural network for conditional flow matching.
    Features:
    - Residual connections for stable training
    - Multi-head attention for long-range dependencies
    - Time embedding and conditioning
    - Adaptive normalization layers
    - Support for sequence-based embeddings
    """
    def __init__(
        self, 
        embedding_dim, 
        seq_length=49,
        hidden_dims=[256, 512, 512, 256], 
        time_embed_dim=128,
        use_attention=True,
        num_heads=8,
        dropout=0.2,
        activation=nn.SiLU(),
        use_layer_norm=True,
        use_skip_connections=True,
        attention_layers=2
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.seq_length = seq_length
        self.time_embed_dim = time_embed_dim
        self.activation = activation
        self.use_attention = use_attention
        self.use_layer_norm = use_layer_norm
        self.use_skip_connections = use_skip_connections
        
        # Time embedding network
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim)
        )
        
        # Context network for time conditioning
        self.time_context_net = nn.Sequential(
            nn.LayerNorm(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(time_embed_dim * 2, time_embed_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(time_embed_dim * 4, time_embed_dim * 2)
        )
        
        # Input processing with normalization
        self.input_layer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]) if use_layer_norm else nn.Identity(),
            activation,
            nn.Dropout(dropout)
        )
        
        # Stack of residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(
                hidden_dims[i], 
                hidden_dims[i+1],
                time_embed_dim,
                dropout=dropout,
                use_layer_norm=use_layer_norm,
                activation=activation
            ) for i in range(len(hidden_dims) - 1)
        ])
        
        # Optional attention mechanism
        if use_attention:
            self.attention_layers = nn.ModuleList([
                AttentionBlock(
                    hidden_dims[-2],
                    num_heads=num_heads,
                    dropout=dropout,
                    use_layer_norm=use_layer_norm
                ) for _ in range(attention_layers)
            ])
        
        # Output network
        output_layers = []
        if use_layer_norm:
            output_layers.append(nn.LayerNorm(hidden_dims[-1]))
        output_layers.extend([
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] // 2, embedding_dim)
        ])
        self.output_layer = nn.Sequential(*output_layers)
        
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, x, t):
        """
        Forward pass with time conditioning and attention.
        Args:
            x: Input tensor [batch_size, seq_length, embedding_dim]
            t: Time tensor [batch_size, seq_length, 1]
        Returns:
            velocity: Predicted velocity field
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Time embedding
        t_emb = self.time_embedding(t)  # [batch_size, seq_len, time_embed_dim]
        context = self.time_context_net(t_emb)  # [batch_size, seq_len, time_embed_dim*2]
        time_scale, time_shift = torch.chunk(context, 2, dim=-1)  # [batch_size, seq_len, time_embed_dim]
        
        # Process each token in the sequence
        h = self.input_layer(x)  # [batch_size, seq_len, hidden_dims[0]]
        
        if self.use_skip_connections and h.shape[-1] == x.shape[-1]:
            h = h + x
        
        for block in self.residual_blocks[:-1]:
            h = block(h, time_scale, time_shift)
        
        if self.use_attention:
            for attn_layer in self.attention_layers:
                h = attn_layer(h)
        
        h = self.residual_blocks[-1](h, time_scale, time_shift)
        return self.output_layer(h)  # [batch_size, seq_len, embedding_dim]


class ResidualBlock(nn.Module):
    """
    Residual block with time conditioning.
    Features:
    - Time-dependent scaling and shifting
    - Dual-path architecture
    - Adaptive normalization
    - Support for sequence-based inputs
    """
    def __init__(self, in_dim, out_dim, time_dim, dropout=0.2, use_layer_norm=True, activation=nn.SiLU()):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, out_dim * 2),
            activation
        )
        
        self.block1 = nn.Sequential(
            nn.LayerNorm(in_dim) if use_layer_norm else nn.Identity(),
            nn.Linear(in_dim, out_dim),
            activation,
            nn.Dropout(dropout)
        )
        
        self.block2 = nn.Sequential(
            nn.LayerNorm(out_dim) if use_layer_norm else nn.Identity(),
            nn.Linear(out_dim, out_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim)
        )
        
        self.shortcut = (nn.Sequential(
            nn.LayerNorm(in_dim) if use_layer_norm else nn.Identity(),
            nn.Linear(in_dim, out_dim)
        ) if in_dim != out_dim else nn.Identity())
    
    def forward(self, x, time_scale, time_shift):
        """
        Forward pass with time conditioning.
        x shape: [batch_size, seq_len, in_dim]
        time_scale, time_shift shape: [batch_size, seq_len, time_dim]
        """
        identity = self.shortcut(x)
        h = self.block1(x)
        
        time_embeddings = self.time_mlp(time_scale)  # [batch_size, seq_len, out_dim*2]
        scale, shift = torch.chunk(time_embeddings, 2, dim=-1)
        
        h = h * (1 + scale) + shift
        
        return self.block2(h) + identity


class AttentionBlock(nn.Module):
    """
    Multi-head attention block.
    Enables learning of long-range dependencies in the input.
    Supports sequence-based inputs.
    """
    def __init__(self, dim, num_heads=8, dropout=0.2, use_layer_norm=True):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        
        self.norm = nn.LayerNorm(dim) if use_layer_norm else nn.Identity()
        self.attention = nn.MultiheadAttention(
            dim, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Apply self-attention with residual connection.
        x shape: [batch_size, seq_len, dim]
        """
        identity = x
        x = self.norm(x)
        attn_out, _ = self.attention(x, x, x)
        return identity + self.dropout(attn_out)


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
    Uses FeedForwardNetwork blocks from EquiformerV2 for core processing.
    """
    def __init__(self,
                 sphere_channels,       # Feature channels (equiv. to embedding_dim)
                 ffn_hidden_channels,   # Hidden channels in FFN blocks
                 time_embed_dim,        # Dimension for time embedding
                 lmax_list,             # List of lmax for each resolution
                 mmax_list,             # List of mmax for each resolution (passed to FFN)
                 SO3_grid,              # SO3_grid for activations (passed to FFN)
                 activation,            # Activation type (passed to FFN)
                 norm_type,             # Normalization type (passed to FFN)
                 use_gate_act,          # Use gate activation? (passed to FFN)
                 use_grid_mlp,          # Use grid MLP? (passed to FFN)
                 use_sep_s2_act,        # Use separable S2 activation? (passed to FFN)
                 num_layers=4,          # Number of equivariant FFN blocks
                 proj_drop=0.0):        # Dropout rate (applied after residual)
        super().__init__()
        self.sphere_channels = sphere_channels
        self.lmax_list = lmax_list
        self.num_resolutions = len(lmax_list)

        # Time embedding network (standard MLP processing invariant time)
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, sphere_channels) # Output matches feature channels
        )

        # Calculate indices corresponding to l=0 components for time conditioning
        # Assumes features are stored as [Node, Coeff, Channel] where Coeff concatenates
        # (res0_l0, res0_l1...), (res1_l0, res1_l1...), ...
        self.l0_indices = []
        offset = 0
        for lmax in self.lmax_list:
            self.l0_indices.append(offset)
            # Number of coefficients for a given lmax is (lmax + 1)**2
            offset += (lmax + 1)**2
        self.total_coefficients = offset

        # Stack of equivariant FeedForwardNetwork blocks
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            # Reuse the FeedForwardNetwork from transformer_block
            # Ensure FeedForwardNetwork is imported where this class is used
            ffn = FeedForwardNetwork(
                sphere_channels=sphere_channels,
                hidden_channels=ffn_hidden_channels,
                output_channels=sphere_channels, # Corrected argument name
                lmax_list=lmax_list,
                mmax_list=mmax_list,
                SO3_grid=SO3_grid,
                activation=activation,
                use_gate_act=use_gate_act,
                use_grid_mlp=use_grid_mlp,
                use_sep_s2_act=use_sep_s2_act,
                # norm_type and proj_drop are handled outside FFN
            )
            self.blocks.append(ffn)

            # Add normalization layer for each block
            # Get max lmax across resolutions for the norm layer
            current_max_lmax = max(lmax_list) if lmax_list else 0
            norm_layer = get_normalization_layer(
                norm_type, lmax=current_max_lmax, num_channels=sphere_channels
            )
            self.norms.append(norm_layer)

        # Dropout layer applied after residual connection
        self.proj_drop = Dropout(proj_drop) if proj_drop > 0. else nn.Identity()

        # Ensure parameters are initialized (optional, inherit from main model apply)


    def forward(self, x_emb, t):
        """
        Forward pass for the equivariant flow model.

        Args:
            x_emb (torch.Tensor): Equivariant features tensor of shape
                                  [num_nodes, num_coefficients, channels].
            t (torch.Tensor): Time tensor of shape [num_nodes, 1].

        Returns:
            torch.Tensor: Predicted velocity field dv/dt, same shape as x_emb.
        """
        # Check input shapes
        num_nodes, num_coeffs, num_channels = x_emb.shape
        assert num_coeffs == self.total_coefficients, f"Input coefficient mismatch: {num_coeffs} vs {self.total_coefficients}"
        assert num_channels == self.sphere_channels, f"Input channel mismatch: {num_channels} vs {self.sphere_channels}"
        assert t.shape == (num_nodes, 1), f"Time tensor shape mismatch: {t.shape} vs ({num_nodes}, 1)"

        # 1. Compute time features (invariant)
        t_feat = self.time_embedding(t) # [num_nodes, sphere_channels]
        # Expand for broadcasting: [num_nodes, 1, sphere_channels]
        t_feat_expanded = t_feat.unsqueeze(1)

        # 2. Apply equivariant blocks with time conditioning and residuals
        h = x_emb
        for i, ffn_block in enumerate(self.blocks):
            identity = h

            # Apply normalization before FFN
            h_norm = self.norms[i](h)

            # Condition with time: Add time features to l=0 components
            h_conditioned = h_norm.clone() # Condition the normalized features
            for l0_idx in self.l0_indices:
                # Add expanded time features to the l=0 slice for all channels
                h_conditioned[:, l0_idx, :] = h_conditioned[:, l0_idx, :] + t_feat_expanded[:, 0, :]

            # Wrap the tensor in SO3_Embedding before passing to FFN
            # Need num_nodes, lmax_list, sphere_channels, device, dtype
            h_so3 = SO3_Embedding(
                length=h_conditioned.shape[0],
                lmax_list=self.lmax_list,
                num_channels=self.sphere_channels,
                device=h_conditioned.device,
                dtype=h_conditioned.dtype
            )
            h_so3.embedding = h_conditioned # Assign the tensor to the object

            # Call FFN with the SO3_Embedding object
            processed_h_so3 = ffn_block(h_so3)

            # Extract the tensor from the result
            processed_h = processed_h_so3.embedding

            # Add residual connection
            h = identity + processed_h # Note: FFN might have internal dropout

            # Apply projection dropout after residual
            h = self.proj_drop(h)

        return h