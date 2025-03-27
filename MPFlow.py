import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
    for better temporal signal representation.
    """
    def __init__(self, dim, max_period=10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
    def forward(self, t):
        """
        Convert time to sinusoidal embedding.
        Args:
            t: Time tensor [batch_size, 1]
        Returns:
            embedding: Time embedding [batch_size, dim]
        """
        device = t.device
        half_dim = self.dim // 2
        
        frequencies = torch.exp(
            torch.linspace(0., np.log(self.max_period), half_dim, device=device)
        )
        
        args = t * frequencies
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1, 0, 0))
            
        return embedding


class EnergyPredictionHead(nn.Module):
    """
    Neural network head for predicting energy values from embeddings.
    
    Args:
        embedding_dim: Dimension of the embedding input
        hidden_dims: List of hidden dimensions for the MLP
        pool_method: Method to aggregate sequence information
    """
    def __init__(self, embedding_dim, seq_length=49, hidden_dims=[256, 128, 64], pool_method='mean'):
        super().__init__()
        self.pool_method = pool_method  # How to aggregate sequence information
        
        layers = []
        input_dim = embedding_dim
        
        # Create MLP layers with ReLU activation and batch normalization
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # Final output layer for scalar energy prediction
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        self.energy_mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass to predict energy.
        
        Args:
            x: Embedding tensor of shape [batch_size, seq_length, embedding_dim]
            
        Returns:
            Predicted energy values of shape [batch_size, 1]
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Pool sequence dimension based on chosen method
        if self.pool_method == 'mean':
            x = x.mean(dim=1)  # [batch_size, embedding_dim]
        elif self.pool_method == 'max':
            x = x.max(dim=1)[0]  # [batch_size, embedding_dim]
        elif self.pool_method == 'sum':
            x = x.sum(dim=1)  # [batch_size, embedding_dim]
        
        return self.energy_mlp(x)