import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from sklearn.decomposition import PCA
from torch.optim.lr_scheduler import OneCycleLR


class ResidualFlowMatchingNetwork(nn.Module):
    """
    Neural network for conditional flow matching with residual connections, 
    normalization, and attention mechanisms.
    """
    def __init__(
        self, 
        embedding_dim, 
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
        self.time_embed_dim = time_embed_dim
        self.activation = activation
        self.use_attention = use_attention
        self.use_layer_norm = use_layer_norm
        self.use_skip_connections = use_skip_connections
        
        # Time embedding with learnable parameters
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim)
        )
        
        # Context network with normalization
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
        
        # Input processing
        self.input_layer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]) if use_layer_norm else nn.Identity(),
            activation,
            nn.Dropout(dropout)
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.residual_blocks.append(
                ResidualBlock(
                    hidden_dims[i], 
                    hidden_dims[i+1],
                    time_embed_dim,
                    dropout=dropout,
                    use_layer_norm=use_layer_norm,
                    activation=activation
                )
            )
        
        # Attention layers
        if use_attention:
            self.attention_layers = nn.ModuleList([
                AttentionBlock(
                    hidden_dims[-2],
                    num_heads=num_heads,
                    dropout=dropout,
                    use_layer_norm=use_layer_norm
                )
                for _ in range(attention_layers)
            ])
        
        # Output processing with multiple layers
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
        # Time embedding with gradient scaling
        t_emb = self.time_embedding(t)
        context = self.time_context_net(t_emb)
        time_scale, time_shift = torch.chunk(context, 2, dim=-1)
        
        # Input processing with residual connection
        h = self.input_layer(x)
        if self.use_skip_connections and h.shape == x.shape:
            h = h + x
        
        # Process through residual blocks
        for block in self.residual_blocks[:-1]:
            h = block(h, time_scale, time_shift)
        
        # Attention layers
        if self.use_attention:
            for attn_layer in self.attention_layers:
                h = attn_layer(h)
        
        # Final residual block
        h = self.residual_blocks[-1](h, time_scale, time_shift)
        
        # Output processing
        velocity = self.output_layer(h)
        
        return velocity


class ResidualBlock(nn.Module):
    """Residual block with additional features"""
    def __init__(self, in_dim, out_dim, time_dim, dropout=0.2, use_layer_norm=True, activation=nn.SiLU()):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        
        # Time conditioning
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, out_dim * 2),
            activation
        )
        
        # Main blocks with residual architecture
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
        
        # Shortcut with normalization
        if in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.LayerNorm(in_dim) if use_layer_norm else nn.Identity(),
                nn.Linear(in_dim, out_dim)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, time_scale, time_shift):
        identity = self.shortcut(x)
        
        # Main path
        h = self.block1(x)
        
        # Time conditioning
        time_embeddings = self.time_mlp(time_scale)
        scale, shift = torch.chunk(time_embeddings, 2, dim=1)
        h = h * (1 + scale) + shift
        
        h = self.block2(h)
        
        return h + identity


class AttentionBlock(nn.Module):
    """Attention block with additional features"""
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
        identity = x
        x = self.norm(x)
        attn_out, _ = self.attention(x, x, x)
        return identity + self.dropout(attn_out)


class SinusoidalTimeEmbedding(nn.Module):
    """Time embedding with extended frequency range."""
    def __init__(self, dim, max_period=10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
    def forward(self, t):
        # t: (batch_size, 1)
        device = t.device
        half_dim = self.dim // 2
        
        # Wider range of frequencies for better signal
        frequencies = torch.exp(
            torch.linspace(
                0., 
                np.log(self.max_period), 
                half_dim, 
                device=device
            )
        )
        
        # Create positions
        args = t * frequencies
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        # Apply additional transformation for better expressivity
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1, 0, 0))
            
        return embedding


class FlowMatching:
    """Manager class for flow matching on message passing trajectories."""
    def __init__(
        self, 
        embedding_dim, 
        hidden_dims=[256, 512, 512, 256], 
        lr=1e-4, 
        flow_matching_type="rectified",  # "standard", "rectified", "stochastic"
        use_attention=True,
        use_adaptive_solver=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.embedding_dim = embedding_dim
        self.flow_matching_type = flow_matching_type
        self.use_adaptive_solver = use_adaptive_solver
        
        # Initialize the residual flow matching network
        self.model = ResidualFlowMatchingNetwork(
            embedding_dim=embedding_dim,
            use_attention=use_attention,
        ).to(device)
        
        # Setup optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=lr, 
            betas=(0.9, 0.999),
            weight_decay=1e-5
        )
        
        # Cosine annealing with warm restarts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=50,  # Restart every 50 epochs
            T_mult=2  # Double the restart interval after each restart
        )
        
        # Set up loss function with spectral normalization option
        self.spectral_norm_weight = 0.01  # Set to 0 to disable
    
    def train_step(self, x0, x1, t):
        """
        Training step with various flow matching options.
        """
        batch_size = x0.shape[0]
        
        # Different types of flow matching
        if self.flow_matching_type == "standard":
            # Standard OT flow matching (straight line)
            ut = x1 - x0
            xt = x0 + t * ut
            
        elif self.flow_matching_type == "rectified":
            # Rectified flow matching (improves sample quality)
            # Use sigmoid-based acceleration of trajectories
            sigmoid_t = torch.sigmoid(5 * (t - 0.5))  # Steeper sigmoid for more pronounced effect
            xt = x0 + sigmoid_t * (x1 - x0)
            ut = x1 - x0  # Target velocity is still the same
            
        elif self.flow_matching_type == "stochastic":
            # Stochastic flow matching - adds noise to the path
            noise_scale = 0.1 * (1.0 - t)  # Noise decreases as t increases
            noise = torch.randn_like(x0) * noise_scale
            xt = x0 + t * (x1 - x0) + noise
            ut = x1 - x0  # Target velocity
        
        # Get the model's prediction
        predicted_ut = self.model(xt, t)
        
        # Flow matching loss 
        loss = F.mse_loss(predicted_ut, ut)
        
        # Optional spectral normalization for stability
        if self.spectral_norm_weight > 0:
            # Calculate Jacobian regularization
            x_detached = xt.clone().detach().requires_grad_(True)
            t_detached = t.clone().detach()
            vector_field = self.model(x_detached, t_detached)
            
            # Get batch gradient
            grad_outputs = torch.ones_like(vector_field)
            gradients = torch.autograd.grad(
                outputs=vector_field,
                inputs=x_detached,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True
            )[0]
            
            # Add Frobenius norm of Jacobian regularization
            spectral_loss = torch.sum(gradients**2) / batch_size
            loss = loss + self.spectral_norm_weight * spectral_loss
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        return loss.item()
    
    def sample_trajectory(self, x0, steps=100, method="dopri5", solver_rtol=1e-5, solver_atol=1e-5):
        """
        Trajectory sampling with modern ODE solvers.
        
        Args:
            x0: Initial state [batch_size, embedding_dim]
            steps: Number of steps (used for fixed step solvers)
            method: "euler", "rk4", "dopri5" (adaptive)
            solver_rtol, solver_atol: Tolerances for adaptive solvers
        """
        self.model.eval()
        
        batch_size = x0.shape[0]
        device = x0.device
        
        # For adaptive solvers, import torchdiffeq
        if method == "dopri5" and self.use_adaptive_solver:
            try:
                from torchdiffeq import odeint
                
                # Define ODE function
                def ode_func(t, x):
                    # Reshape x if needed for batch processing
                    x_reshaped = x.view(batch_size, -1)
                    t_tensor = torch.ones((batch_size, 1), device=device) * t
                    return self.model(x_reshaped, t_tensor).view(-1)
                
                # Initial state
                x0_flat = x0.reshape(-1)
                
                # Integration times
                t_span = torch.linspace(0, 1, steps+1, device=device)
                
                # Solve ODE with adaptive solver
                trajectory_flat = odeint(
                    ode_func, 
                    x0_flat, 
                    t_span, 
                    method=method,
                    rtol=solver_rtol, 
                    atol=solver_atol
                )
                
                # Reshape to original dimensions
                trajectory = trajectory_flat.view(-1, batch_size, self.embedding_dim).permute(1, 0, 2)
                return trajectory
                
            except ImportError:
                print("torchdiffeq not installed, falling back to RK4")
                method = "rk4"
        
        # Fixed step solvers (existing implementation)
        ts = torch.linspace(0, 1, steps+1, device=device)
        dt = 1.0 / steps
        
        trajectory = torch.zeros((batch_size, steps+1, self.embedding_dim), device=device)
        trajectory[:, 0] = x0
        
        x = x0.clone()
        
        with torch.no_grad():
            for i in range(steps):
                t = ts[i] * torch.ones((batch_size, 1), device=device)
                
                if method == "euler":
                    # Euler integration
                    velocity = self.model(x, t)
                    x = x + dt * velocity
                    
                elif method == "rk4":
                    # 4th order Runge-Kutta
                    k1 = self.model(x, t)
                    k2 = self.model(x + dt * k1 / 2, t + dt / 2)
                    k3 = self.model(x + dt * k2 / 2, t + dt / 2)
                    k4 = self.model(x + dt * k3, t + dt)
                    
                    x = x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
                
                trajectory[:, i+1] = x
        
        return trajectory

    @torch.no_grad()
    def visualize(self, trajectories, step=50, output_dir='flow_output', plots=["2d", "3d", "tsne"]):
        """
        Visualization with multiple plotting options and statistical analysis
        
        Args:
            trajectories: Ground truth trajectories
            step: Number of steps for sampled trajectories
            output_dir: Directory to save outputs
            plots: List of plot types to create
        """
        # convert trajectories to torch tensor
        trajectories = torch.tensor(trajectories, dtype=torch.float32, device=self.device)
        
        # Select samples to visualize
        viz_samples = min(12, len(trajectories))
        indices = np.random.choice(len(trajectories), viz_samples, replace=False)
        
        # Get ground truth trajectories for selected indices
        ground_truth = trajectories[indices]
        n_timesteps = ground_truth.shape[1]
        
        # Sample trajectories with same number of steps as ground truth
        x0 = ground_truth[:, 0].to(self.device)
        x1 = ground_truth[:, -1].to(self.device)  # For reference only
        
        # Generate sampled trajectories with same number of steps as ground truth
        sampled_trajectories = self.sample_trajectory(x0, steps=n_timesteps-1, method="dopri5")
        
        # Calculate statistics
        mse_per_sample = torch.mean((ground_truth.cpu() - sampled_trajectories.cpu())**2, dim=(1, 2))
        avg_mse = mse_per_sample.mean().item()
        
        # Directory for this visualization run
        os.makedirs(output_dir, exist_ok=True)
        
        # Save statistics to file
        with open(os.path.join(output_dir, f"trajectory_stats_{step}.txt"), "w") as f:
            f.write(f"Average MSE: {avg_mse:.6f}\n")
            f.write(f"Per-sample MSE: {mse_per_sample.numpy()}\n")
        
        # Create visualizations
        if "2d" in plots:
            self._create_2d_pca_plot(ground_truth, sampled_trajectories, step, output_dir)
        
        if "3d" in plots:
            self._create_3d_pca_plot(ground_truth, sampled_trajectories, step, output_dir)
        
        if "tsne" in plots:
            self._create_tsne_plot(ground_truth, sampled_trajectories, step, output_dir)

    def _create_2d_pca_plot(self, ground_truth, sampled_trajectories, step, output_dir):
        """Create 2D PCA visualization of trajectories"""
        # Combine for consistent PCA
        combined = torch.cat([ground_truth.cpu(), sampled_trajectories.cpu()], dim=0)
        combined_flat = combined.reshape(-1, combined.shape[-1])
        
        # Apply PCA
        pca = PCA(n_components=2)
        combined_2d = pca.fit_transform(combined_flat.numpy())
        
        # Split back
        n_samples = ground_truth.shape[0]
        gt_len = ground_truth.shape[1]
        sampled_len = sampled_trajectories.shape[1]
        
        gt_2d = combined_2d[:n_samples * gt_len].reshape(n_samples, gt_len, 2)
        sampled_2d = combined_2d[n_samples * gt_len:].reshape(n_samples, sampled_len, 2)
        
        # Create plot
        plt.figure(figsize=(12, 10))
        colors = plt.cm.tab20(np.linspace(0, 1, n_samples))
        
        for i in range(n_samples):
            # Ground truth
            plt.plot(
                gt_2d[i, :, 0], 
                gt_2d[i, :, 1], 
                '--', 
                color=colors[i], 
                alpha=0.7, 
                linewidth=2,
                label=f'GT {i+1}' if i == 0 else None
            )
            
            # Sampled
            plt.plot(
                sampled_2d[i, :, 0], 
                sampled_2d[i, :, 1], 
                '-', 
                color=colors[i], 
                alpha=0.9,
                linewidth=2,
                label=f'Sampled {i+1}' if i == 0 else None
            )
            
            # Mark start and end
            plt.scatter(gt_2d[i, 0, 0], gt_2d[i, 0, 1], color='blue', s=100, marker='o')
            plt.scatter(gt_2d[i, -1, 0], gt_2d[i, -1, 1], color='red', s=100, marker='o')
        
        plt.title('Flow Trajectories - PCA Projection')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        plt.grid(True, alpha=0.3)
        
        # Only show first pair in legend
        plt.legend(["Ground Truth", "Sampled"], loc='upper right')
        
        plt.savefig(os.path.join(output_dir, f"flow_trajectories_2d_{step}.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def _create_3d_pca_plot(self, ground_truth, sampled_trajectories, step, output_dir):
        """Create 3D PCA visualization of trajectories"""
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            # Combine for consistent PCA
            combined = torch.cat([ground_truth.cpu(), sampled_trajectories.cpu()], dim=0)
            combined_flat = combined.reshape(-1, combined.shape[-1])
            
            # Apply PCA for 3D
            pca = PCA(n_components=3)
            combined_3d = pca.fit_transform(combined_flat.numpy())
            
            # Split back
            n_samples = ground_truth.shape[0]
            gt_len = ground_truth.shape[1]
            sampled_len = sampled_trajectories.shape[1]
            
            gt_3d = combined_3d[:n_samples * gt_len].reshape(n_samples, gt_len, 3)
            sampled_3d = combined_3d[n_samples * gt_len:].reshape(n_samples, sampled_len, 3)
            
            # Create plot
            fig = plt.figure(figsize=(14, 12))
            ax = fig.add_subplot(111, projection='3d')
            colors = plt.cm.tab20(np.linspace(0, 1, n_samples))
            
            for i in range(n_samples):
                # Ground truth
                ax.plot(
                    gt_3d[i, :, 0], 
                    gt_3d[i, :, 1], 
                    gt_3d[i, :, 2],
                    '--', 
                    color=colors[i], 
                    alpha=0.7,
                    linewidth=2
                )
                
                # Sampled
                ax.plot(
                    sampled_3d[i, :, 0], 
                    sampled_3d[i, :, 1], 
                    sampled_3d[i, :, 2],
                    '-', 
                    color=colors[i], 
                    alpha=0.9,
                    linewidth=2
                )
                
                # Mark start and end
                ax.scatter(
                    gt_3d[i, 0, 0], 
                    gt_3d[i, 0, 1], 
                    gt_3d[i, 0, 2], 
                    color='blue', 
                    s=100, 
                    marker='o'
                )
                ax.scatter(
                    gt_3d[i, -1, 0], 
                    gt_3d[i, -1, 1], 
                    gt_3d[i, -1, 2], 
                    color='red', 
                    s=100, 
                    marker='o'
                )
            
            ax.set_title('Flow Trajectories - 3D PCA Projection')
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
            
            # Create custom legend
            from matplotlib.lines import Line2D
            custom_lines = [
                Line2D([0], [0], linestyle='--', color='gray', linewidth=2),
                Line2D([0], [0], linestyle='-', color='gray', linewidth=2)
            ]
            ax.legend(custom_lines, ['Ground Truth', 'Sampled'])
            
            plt.savefig(os.path.join(output_dir, f"flow_trajectories_3d_{step}.png"), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error creating 3D plot: {e}")

    def _create_tsne_plot(self, ground_truth, sampled_trajectories, step, output_dir):
        """Create t-SNE visualization of trajectories"""
        try:
            from sklearn.manifold import TSNE
            
            # Compute t-SNE on flattened data
            gt_flat = ground_truth.cpu().reshape(-1, ground_truth.shape[-1]).numpy()
            sampled_flat = sampled_trajectories.cpu().reshape(-1, sampled_trajectories.shape[-1]).numpy()
            
            # Take a subset for t-SNE if data is large
            max_points = 5000
            if gt_flat.shape[0] > max_points:
                indices = np.random.choice(gt_flat.shape[0], max_points, replace=False)
                gt_flat = gt_flat[indices]
                sampled_flat = sampled_flat[indices]
            
            # Combine for consistent transformation
            combined = np.vstack([gt_flat, sampled_flat])
            
            # Apply t-SNE
            tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
            combined_tsne = tsne.fit_transform(combined)
            
            # Split back
            gt_tsne = combined_tsne[:gt_flat.shape[0]]
            sampled_tsne = combined_tsne[gt_flat.shape[0]:]
            
            # Plot
            plt.figure(figsize=(12, 10))
            
            plt.scatter(
                gt_tsne[:, 0], 
                gt_tsne[:, 1], 
                alpha=0.6, 
                label='Ground Truth',
                s=20
            )
            
            plt.scatter(
                sampled_tsne[:, 0], 
                sampled_tsne[:, 1], 
                alpha=0.6, 
                label='Sampled',
                s=20
            )
            
            plt.title('t-SNE Visualization of Embeddings')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.savefig(os.path.join(output_dir, f"tsne_visualization_{step}.png"), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error creating t-SNE plot: {e}")

    def save_model(self, path):
        """Save the model and optimizer state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path):
        """Load the model and optimizer state."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
# Example usage
def prepare_data(trajectories, train_ratio=0.9):
    """
    Prepare training and validation data.
    
    Args:
        trajectories: NumPy array of shape [n_samples, n_layers, embedding_dim]
        train_ratio: Ratio of data to use for training
        
    Returns:
        train_data, val_data: Training and validation datasets
    """
    n_samples = trajectories.shape[0]
    n_train = int(n_samples * train_ratio)
    
    train_data = trajectories[:n_train]
    val_data = trajectories[n_train:]
    
    return train_data, val_data

def train_flow_model(
    trajectories, 
    embedding_dim, 
    num_epochs=100, 
    batch_size=64, 
    output_dir='flow_output',
    flow_matching_type="rectified",
    validation_interval=10
):
    """
    Train a flow matching model.
    
    Args:
        trajectories: NumPy array of shape [n_samples, n_layers, embedding_dim]
        embedding_dim: Dimension of the embeddings
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        output_dir: Directory to save outputs
        flow_matching_type: Type of flow matching to use
        validation_interval: Interval for validation checks
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data with stratified split
    train_data, val_data = prepare_data(trajectories, train_ratio=0.8)
    
    # Initialize model with residual architecture
    flow_model = FlowMatching(
        embedding_dim=embedding_dim,
        hidden_dims=[256, 512, 768, 512, 256],  # Deeper network
        lr=2e-4,  # Slightly higher LR with cosine schedule
        flow_matching_type=flow_matching_type,
        use_attention=True,
        use_adaptive_solver=True
    )
    # print(f"Model architecture: {flow_model.model}")
    print(f"Model parameters: {sum(p.numel() for p in flow_model.model.parameters())}")
    
    # Try to load existing model if available
    model_path = os.path.join(output_dir, 'flow_matching_model.pt')
    if os.path.exists(model_path):
        try:
            flow_model.load_model(model_path)
            print("Loaded existing model for continued training.")
        except:
            print("Could not load existing model, starting fresh.")
    
    # Setup for training
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 30  # For early stopping
    patience_counter = 0
    
    # Training loop with progress bar
    for epoch in tqdm(range(num_epochs)):
        epoch_losses = []
        
        # Process in batches
        num_batches = len(train_data) // batch_size
        for i in range(num_batches):
            # Get batch
            batch_indices = np.random.choice(len(train_data), batch_size, replace=False)
            batch_data = train_data[batch_indices]
            
            # Convert to torch tensors
            batch_tensor = torch.tensor(batch_data, dtype=torch.float32, device=flow_model.device)
            
            # Get start and end states
            x0 = batch_tensor[:, 0]  # First layer
            x1 = batch_tensor[:, -1]  # Last layer
            
            # Sample random times for flow matching
            t = torch.rand(batch_size, 1, device=flow_model.device)
            
            # Train step
            loss = flow_model.train_step(x0, x1, t)
            epoch_losses.append(loss)
        
        # Calculate average loss
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(avg_loss)
        
        # Log training progress
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        # Validation at intervals
        if (epoch + 1) % validation_interval == 0 or epoch == num_epochs - 1:
            val_loss = validate_model(flow_model, val_data, batch_size)
            val_losses.append(val_loss)
            
            print(f"Validation Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                flow_model.save_model(os.path.join(output_dir, 'best_flow_model.pt'))
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break
        
        # Learning rate scheduler step
        flow_model.scheduler.step()
        
        # Visualization at intervals
        if epoch % 50 == 0 or epoch == num_epochs - 1:
            flow_model.visualize(
                val_data[:50],  # Use subset of validation data
                step=val_data.shape[1]-1,
                output_dir=output_dir,
                plots=["2d", "3d"]
            )
    
    # Save final model
    flow_model.save_model(os.path.join(output_dir, 'flow_matching_model.pt'))
    
    # Create separate plots for training and validation losses
    # Training loss plot
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Progress')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'training_loss_curve.png'))
    plt.close()
    
    # Validation loss plot
    plt.figure(figsize=(12, 6))
    val_epochs = list(range(0, num_epochs, validation_interval))
    if len(val_losses) > len(val_epochs):
        val_epochs.append(num_epochs - 1)
    plt.plot(val_epochs[:len(val_losses)], val_losses, 'r-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss Progress')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'validation_loss_curve.png'))
    plt.close()
    
    # Load best model for return
    flow_model.load_model(os.path.join(output_dir, 'best_flow_model.pt'))
    
    return flow_model, {'train': train_losses, 'val': val_losses}


def validate_model(flow_model, val_data, batch_size=64):
    """
    Validate the flow matching model on validation data.
    
    Args:
        flow_model: Trained flow matching model
        val_data: Validation data
        batch_size: Batch size for validation
        
    Returns:
        val_loss: Validation loss
    """
    flow_model.model.eval()
    val_losses = []
    
    with torch.no_grad():
        # Process in batches
        num_batches = len(val_data) // batch_size + (1 if len(val_data) % batch_size != 0 else 0)
        
        for i in range(num_batches):
            # Get batch
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(val_data))
            batch_data = val_data[start_idx:end_idx]
            
            # Convert to torch tensors
            batch_tensor = torch.tensor(batch_data, dtype=torch.float32, device=flow_model.device)
            
            # Get start and end states
            x0 = batch_tensor[:, 0]  # First layer
            x1 = batch_tensor[:, -1]  # Last layer
            
            # Sample times - use fixed grid for validation
            batch_size_actual = x0.shape[0]
            t_values = torch.linspace(0.1, 0.9, 5).repeat(batch_size_actual, 1).to(flow_model.device)
            
            for t_idx in range(t_values.shape[1]):
                t = t_values[:, t_idx:t_idx+1]
                
                # Straight-line interpolation
                ut = x1 - x0  # Target velocity
                xt = x0 + t * ut  # Current point
                
                # Get model prediction
                predicted_ut = flow_model.model(xt, t)
                
                # Calculate MSE loss
                loss = F.mse_loss(predicted_ut, ut)
                val_losses.append(loss.item())
    
    # Return average validation loss
    return sum(val_losses) / len(val_losses)

if __name__ == "__main__":
    # Load the NPZ file
    output_dir = 'flow_output'
    os.makedirs(output_dir, exist_ok=True)
    
    data = np.load('logs/2316385/s2ef_predictions.npz')
    trajectories = data['latents'].reshape(-1, 21, 128)
    
    print(f"Loaded trajectories shape: {trajectories.shape}")
    
    # Train the residual flow matching model
    embedding_dim = 128  # Matches your latent dimension
    num_epochs = 500
    batch_size = 64
    
    # Choose flow matching type
    # "standard" - OT/straight line
    # "rectified" - Better sample quality with sigmoid acceleration
    # "stochastic" - Adds noise to paths, good for exploration
    flow_matching_type = "rectified"
    
    # Train flow matching model
    flow_model, losses = train_flow_model(
        trajectories=trajectories,
        embedding_dim=embedding_dim,
        num_epochs=num_epochs,
        batch_size=batch_size,
        flow_matching_type=flow_matching_type,
        output_dir=output_dir
    )
    
    # Create comprehensive visualizations
    val_data = prepare_data(trajectories)[1]  # Get validation set
    flow_model.visualize(
        val_data[:20],
        step=val_data.shape[1]-1,
        output_dir=output_dir,
        plots=["2d", "3d", "tsne"]
    )
    
    print("Training completed and model saved!")