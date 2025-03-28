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
from MPFlow import MPFlow
    
    
class FlowMatching:
    """
    A class to manage flow matching on message passing trajectories.
    Implements training, sampling, and visualization of flow-based trajectories.
    
    Key features:
    - Supports multiple sampling methods (euler, rk4)
    - Includes visualization tools (2D PCA, 3D PCA, t-SNE)
    - Implements cosine annealing with warm restarts
    
    Args:
        embedding_dim: Dimension of the embedding
        hidden_dims: List of hidden dimensions
        lr: Learning rate
        use_attention: Whether to use attention
        use_adaptive_solver: Whether to use an adaptive solver
        device: Device to run the model on
    """
    def __init__(
        self, 
        embedding_dim, 
        hidden_dims=[256, 512, 512, 256], 
        lr=1e-4, 
        use_attention=True,
        use_adaptive_solver=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.embedding_dim = embedding_dim
        self.use_adaptive_solver = use_adaptive_solver
        
        # Initialize the residual flow matching network
        self.model = MPFlow(
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
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
    
    def train_step(self, x0, x1, t):
        """
        Performs a single training step using flow matching loss.
        Combines MSE loss with directional loss for better trajectory learning.
        Loss = MSE + 0.005 * DirectionalLoss
        """
        batch_size = x0.shape[0]
        
        ut = x1 - x0
        xt = x0 + t * ut
            
        # Get the model's prediction
        predicted_ut = self.model(xt, t)
        
        # Flow matching loss 
        mse_loss = F.mse_loss(predicted_ut, ut)
        direction_loss = (1 - torch.cosine_similarity(predicted_ut, ut, dim=1)).mean()
        loss = mse_loss + direction_loss * 0.005
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        return loss.item()
    
    def sample_trajectory(self, x0, steps=2, method="rk4", solver_rtol=1e-5, solver_atol=1e-5):
        """
        Samples a trajectory from start to end points.
        Supports multiple integration methods:
        - euler: Simple first-order method
        - rk4: 4th order Runge-Kutta (more accurate)
        Returns only start and end points for efficiency.
        """
        self.model.eval()
        batch_size = x0.shape[0]
        device = x0.device

        # For fixed step methods, just compute the final point
        x = x0.clone()
        
        with torch.no_grad():
            if method == "euler":
                # Single large step
                t = torch.ones((batch_size, 1), device=device)
                velocity = self.model(x, t)
                x = x + velocity  # Single step from 0 to 1
                
            elif method == "rk4":
                # Single RK4 step from t=0 to t=1
                t = torch.zeros((batch_size, 1), device=device)
                k1 = self.model(x, t)
                k2 = self.model(x + 0.5 * k1, t + 0.5)
                k3 = self.model(x + 0.5 * k2, t + 0.5)
                k4 = self.model(x + k3, t + 1.0)
                
                x = x + (1.0 / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Return start and end points only
        trajectory = torch.stack([x0, x], dim=1)  # [batch, 2, dim]
        return trajectory

    def visualize(self, trajectories, step=100, output_dir='flow_output', plots=["2d", "3d", "tsne"]):
        """
        Creates comprehensive visualizations of trajectories.
        Generates multiple plot types:
        - 2D PCA: Principal component projection
        - 3D PCA: 3D visualization of main components
        - t-SNE: Non-linear dimensionality reduction
        Also computes and saves trajectory statistics.
        """
        # convert trajectories to torch tensor
        trajectories = torch.tensor(trajectories, dtype=torch.float32, device=self.device)
        
        # Select samples to visualize
        viz_samples = min(8, len(trajectories))
        indices = np.arange(viz_samples)  # Use first viz_samples trajectories
        
        # Get ground truth trajectories for selected indices
        ground_truth = trajectories[indices]
        
        # Sample trajectories with fixed number of steps
        x0 = ground_truth[:, 0].to(self.device)
        x1 = ground_truth[:, -1].to(self.device)  # For reference only
        x1_gt = ground_truth[:, -1].to(self.device)  # Ground truth final state
        
        # Use fixed number of steps (e.g., 100) instead of matching ground truth
        sampled_trajectories = self.sample_trajectory(x0, steps=step, method="rk4")
        
        # For comparison with ground truth, we can interpolate if needed
        # This step is optional and only needed if you want to compute exact MSE
        if ground_truth.shape[1] != sampled_trajectories.shape[1]:
            # Interpolate sampled trajectories to match ground truth timesteps
            sampled_interp = F.interpolate(
                sampled_trajectories.permute(0, 2, 1),  # [batch, embedding_dim, time]
                size=ground_truth.shape[1],
                mode='linear',
                align_corners=True
            ).permute(0, 2, 1)  # [batch, time, embedding_dim]
            
            # Calculate MSE using interpolated trajectories
            mse_per_sample = torch.mean((x1_gt.cpu() - sampled_interp[:, -1, :].cpu())**2, dim=1)
        else:
            mse_per_sample = torch.mean((x1_gt.cpu() - sampled_trajectories[:, -1, :].cpu())**2, dim=1)
        
        # Calculate statistics
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
        """
        Creates 2D PCA visualization comparing ground truth vs sampled trajectories.
        Shows both full ground truth paths and sampled endpoints.
        Includes variance ratios for interpretability.
        """
        # Combine for consistent PCA - need to handle different timesteps
        gt_flat = ground_truth.cpu().reshape(-1, ground_truth.shape[-1])
        sampled_flat = sampled_trajectories.cpu().reshape(-1, sampled_trajectories.shape[-1])
        
        # Apply PCA on all points
        pca = PCA(n_components=2)
        gt_2d = pca.fit_transform(gt_flat.numpy())
        sampled_2d = pca.transform(sampled_flat.numpy())
        
        # Reshape back
        n_samples = ground_truth.shape[0]
        gt_timesteps = ground_truth.shape[1]  # 21 steps
        sampled_timesteps = sampled_trajectories.shape[1]  # 2 steps (start/end)
        
        gt_2d = gt_2d.reshape(n_samples, gt_timesteps, 2)
        sampled_2d = sampled_2d.reshape(n_samples, sampled_timesteps, 2)
        
        # Create plot
        plt.figure(figsize=(12, 10))
        colors = plt.cm.tab20(np.linspace(0, 1, n_samples))
        
        for i in range(n_samples):
            # Ground truth - full trajectory
            plt.plot(
                gt_2d[i, :, 0], 
                gt_2d[i, :, 1], 
                '--', 
                color=colors[i], 
                alpha=0.7, 
                linewidth=2,
                label=f'GT {i+1}' if i == 0 else None
            )
            
            # Sampled - start to end only
            plt.plot(
                sampled_2d[i, :, 0], 
                sampled_2d[i, :, 1], 
                '-', 
                color=colors[i], 
                alpha=0.9,
                linewidth=2,
                label=f'Sampled {i+1}' if i == 0 else None
            )
            
            # Mark start and end points (all circles)
            plt.scatter(gt_2d[i, 0, 0], gt_2d[i, 0, 1], color='blue', s=50, marker='o')
            plt.scatter(gt_2d[i, -1, 0], gt_2d[i, -1, 1], color='red', s=50, marker='o')
            plt.scatter(sampled_2d[i, 0, 0], sampled_2d[i, 0, 1], color='blue', s=50, marker='o')
            plt.scatter(sampled_2d[i, -1, 0], sampled_2d[i, -1, 1], color='black', s=25, marker='x')
        
        plt.title('Flow Trajectories - PCA Projection\n(Full GT vs Sampled Start/End)')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        plt.grid(True, alpha=0.3)
        
        plt.legend([
            "Ground Truth (21 steps)", 
            "Sampled (start/end)",
            "Start Points",
            "End Points"
        ], loc='upper right')
        
        plt.savefig(os.path.join(output_dir, f"flow_trajectories_2d_{step}.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def _create_3d_pca_plot(self, ground_truth, sampled_trajectories, step, output_dir):
        """
        Generates 3D PCA plots with interactive viewing capabilities.
        Visualizes trajectories in 3D space with custom legends and markers.
        Handles potential 3D plotting errors gracefully.
        """
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            # Apply PCA on all points
            gt_flat = ground_truth.cpu().reshape(-1, ground_truth.shape[-1])
            sampled_flat = sampled_trajectories.cpu().reshape(-1, sampled_trajectories.shape[-1])
            
            # Apply PCA for 3D
            pca = PCA(n_components=3)
            gt_3d = pca.fit_transform(gt_flat.numpy())
            sampled_3d = pca.transform(sampled_flat.numpy())
            
            # Reshape back
            n_samples = ground_truth.shape[0]
            gt_timesteps = ground_truth.shape[1]  # 21 steps
            sampled_timesteps = sampled_trajectories.shape[1]  # 2 steps
            
            gt_3d = gt_3d.reshape(n_samples, gt_timesteps, 3)
            sampled_3d = sampled_3d.reshape(n_samples, sampled_timesteps, 3)
            
            # Create plot
            fig = plt.figure(figsize=(14, 12))
            ax = fig.add_subplot(111, projection='3d')
            colors = plt.cm.tab20(np.linspace(0, 1, n_samples))
            
            for i in range(n_samples):
                # Ground truth - full trajectory
                ax.plot(
                    gt_3d[i, :, 0], 
                    gt_3d[i, :, 1], 
                    gt_3d[i, :, 2],
                    '--', 
                    color=colors[i], 
                    alpha=0.7,
                    linewidth=2
                )
                
                # Sampled - start to end only
                ax.plot(
                    sampled_3d[i, :, 0], 
                    sampled_3d[i, :, 1], 
                    sampled_3d[i, :, 2],
                    '-', 
                    color=colors[i], 
                    alpha=0.9,
                    linewidth=2
                )
                
                # Mark all points with circles
                ax.scatter(
                    gt_3d[i, 0, 0], 
                    gt_3d[i, 0, 1], 
                    gt_3d[i, 0, 2], 
                    color='blue', 
                    s=50, 
                    marker='o'
                )
                ax.scatter(
                    gt_3d[i, -1, 0], 
                    gt_3d[i, -1, 1], 
                    gt_3d[i, -1, 2], 
                    color='red', 
                    s=50, 
                    marker='o'
                )
                ax.scatter(
                    sampled_3d[i, 0, 0], 
                    sampled_3d[i, 0, 1], 
                    sampled_3d[i, 0, 2], 
                    color='blue', 
                    s=50, 
                    marker='o'
                )
                ax.scatter(
                    sampled_3d[i, -1, 0], 
                    sampled_3d[i, -1, 1], 
                    sampled_3d[i, -1, 2], 
                    color='black', 
                    s=25, 
                    marker='x'
                )
            
            ax.set_title('Flow Trajectories - 3D PCA Projection\n(Full GT vs Sampled Start/End)')
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
            
            # Create custom legend
            from matplotlib.lines import Line2D
            custom_lines = [
                Line2D([0], [0], linestyle='--', color='gray', linewidth=2),
                Line2D([0], [0], linestyle='-', color='gray', linewidth=2),
                Line2D([0], [0], linestyle='none', marker='o', color='blue', markersize=10),
                Line2D([0], [0], linestyle='none', marker='o', color='red', markersize=10)
            ]
            ax.legend(custom_lines, [
                'Ground Truth (21 steps)', 
                'Sampled (start/end)',
                'Start Points',
                'End Points'
            ])
            
            plt.savefig(os.path.join(output_dir, f"flow_trajectories_3d_{step}.png"), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error creating 3D plot: {e}", flush=True)

    def _create_tsne_plot(self, ground_truth, sampled_trajectories, step, output_dir):
        """
        Creates t-SNE visualization for non-linear trajectory analysis.
        Automatically handles large datasets by sampling.
        Maintains temporal consistency in the visualization.
        """
        try:
            from sklearn.manifold import TSNE
            
            # Compute t-SNE on flattened data
            gt_flat = ground_truth.cpu().reshape(-1, ground_truth.shape[-1]).numpy()
            sampled_flat = sampled_trajectories.cpu().reshape(-1, sampled_trajectories.shape[-1]).numpy()
            
            # Take a subset for t-SNE if data is large
            max_points = 5000
            n_samples = ground_truth.shape[0]
            gt_timesteps = ground_truth.shape[1]
            sampled_timesteps = sampled_trajectories.shape[1]
            
            if gt_flat.shape[0] > max_points:
                # Sample trajectories, but keep all timesteps for each selected trajectory
                n_trajectories = max_points // gt_timesteps
                selected_indices = np.random.choice(n_samples, n_trajectories, replace=False)
                gt_flat = gt_flat.reshape(n_samples, gt_timesteps, -1)[selected_indices].reshape(-1, ground_truth.shape[-1])
                sampled_flat = sampled_flat.reshape(n_samples, sampled_timesteps, -1)[selected_indices].reshape(-1, sampled_trajectories.shape[-1])
                n_samples = n_trajectories
            
            # Combine for consistent transformation
            combined = np.vstack([gt_flat, sampled_flat])
            
            # Apply t-SNE
            tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
            combined_tsne = tsne.fit_transform(combined)
            
            # Split back
            gt_tsne = combined_tsne[:gt_flat.shape[0]].reshape(n_samples, gt_timesteps, 2)
            sampled_tsne = combined_tsne[gt_flat.shape[0]:].reshape(n_samples, sampled_timesteps, 2)
            
            # Plot
            plt.figure(figsize=(12, 10))
            colors = plt.cm.tab20(np.linspace(0, 1, n_samples))
            
            for i in range(n_samples):
                # Ground truth - full trajectory
                plt.plot(
                    gt_tsne[i, :, 0],
                    gt_tsne[i, :, 1],
                    '--',
                    color=colors[i],
                    alpha=0.7,
                    linewidth=2
                )
                
                # Sampled - start to end only
                plt.plot(
                    sampled_tsne[i, :, 0],
                    sampled_tsne[i, :, 1],
                    '-',
                    color=colors[i],
                    alpha=0.9,
                    linewidth=2
                )
                
                # Mark all points with circles
                plt.scatter(gt_tsne[i, 0, 0], gt_tsne[i, 0, 1], color='blue', s=50, marker='o')
                plt.scatter(gt_tsne[i, -1, 0], gt_tsne[i, -1, 1], color='red', s=50, marker='o')
                plt.scatter(sampled_tsne[i, 0, 0], sampled_tsne[i, 0, 1], color='blue', s=50, marker='o')
                plt.scatter(sampled_tsne[i, -1, 0], sampled_tsne[i, -1, 1], color='black', s=25, marker='x')
            
            plt.title('t-SNE Visualization\n(Full GT vs Sampled Start/End)')
            plt.grid(True, alpha=0.3)
            
            plt.legend([
                "Ground Truth (21 steps)",
                "Sampled (start/end)",
                "Start Points",
                "End Points"
            ], loc='upper right')
            
            plt.savefig(os.path.join(output_dir, f"tsne_visualization_{step}.png"), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error creating t-SNE plot: {e}", flush=True)

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
def prepare_data(trajectories, train_ratio=0.8):
    """
    Splits trajectory data into training and validation sets.
    Simple split without shuffling to maintain temporal order.
    Returns: (train_data, val_data) tuple
    """
    n_samples = trajectories.shape[0]
    n_train = int(n_samples * train_ratio)
    
    train_data = trajectories[:n_train]
    val_data = trajectories[n_train:]
    
    return train_data, val_data

def train_flow_model(
    trajectories, 
    embedding_dim, 
    num_epochs=1000, 
    batch_size=32, 
    output_dir='flow_output',
    validation_interval=10,
    hidden_dims=[256, 512, 768, 512, 256]
):
    """
    Main training loop for the flow matching model.
    Features:
    - Early stopping with patience
    - Model checkpointing
    - Progress visualization
    - Learning rate scheduling
    - Validation monitoring
    Returns: (trained_model, loss_history)
    """
    os.makedirs(output_dir, exist_ok=True)
    train_data, val_data = prepare_data(trajectories, train_ratio=0.8)
    
    flow_model = FlowMatching(
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims,
        lr=1e-4,
        use_attention=True,
        use_adaptive_solver=True
    )
    print(f"Model parameters: {sum(p.numel() for p in flow_model.model.parameters())}", flush=True)
    
    model_path = os.path.join(output_dir, 'flow_matching_model.pt')
    if os.path.exists(model_path):
        try:
            flow_model.load_model(model_path)
            print("Loaded existing model for continued training.", flush=True)
        except:
            print("Could not load existing model, starting fresh.", flush=True)
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience, patience_counter = 100, 0
    
    for epoch in tqdm(range(num_epochs)):
        epoch_losses = []
        num_batches = len(train_data) // batch_size
        
        for i in range(num_batches):
            batch_indices = np.random.choice(len(train_data), batch_size, replace=False)
            batch_tensor = torch.tensor(train_data[batch_indices], dtype=torch.float32, device=flow_model.device)
            
            x0, x1 = batch_tensor[:, 0], batch_tensor[:, -1]
            t = torch.rand(batch_size, 1, device=flow_model.device)
            
            loss = flow_model.train_step(x0, x1, t)
            epoch_losses.append(loss)
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}", flush=True)
        
        if (epoch + 1) % validation_interval == 0 or epoch == num_epochs - 1:
            val_loss = validate_model(flow_model, val_data, batch_size)
            val_losses.append(val_loss)
            print(f"Validation Loss: {val_loss:.6f}", flush=True)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                flow_model.save_model(os.path.join(output_dir, 'best_flow_model.pt'))
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs", flush=True)
                break
        
        flow_model.scheduler.step()
        
        if (epoch + 1) % 50 == 0 or epoch == num_epochs - 1:
            flow_model.visualize(
                val_data[:50],
                step=val_data.shape[1]-1,
                output_dir=output_dir,
                plots=["2d", "3d"]
            )
    
    flow_model.save_model(os.path.join(output_dir, 'flow_matching_model.pt'))
    
    # Plot training loss
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, 'b-', linewidth=2)
    plt.xlabel('Epoch'), plt.ylabel('Loss')
    plt.title('Training Loss Progress')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'training_loss_curve.png'))
    plt.close()
    
    # Plot validation loss
    plt.figure(figsize=(12, 6))
    val_epochs = list(range(0, num_epochs, validation_interval))
    if len(val_losses) > len(val_epochs):
        val_epochs.append(num_epochs - 1)
    plt.plot(val_epochs[:len(val_losses)], val_losses, 'r-', linewidth=2)
    plt.xlabel('Epoch'), plt.ylabel('Loss')
    plt.title('Validation Loss Progress')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'validation_loss_curve.png'))
    plt.close()
    
    flow_model.load_model(os.path.join(output_dir, 'best_flow_model.pt'))
    return flow_model, {'train': train_losses, 'val': val_losses}

def validate_model(flow_model, val_data, batch_size=32):
    """
    Evaluates model performance on validation data.
    Uses multiple time points for robust evaluation.
    Returns average loss across all validation samples.
    """
    flow_model.model.eval()
    val_losses = []
    
    with torch.no_grad():
        num_batches = len(val_data) // batch_size + (1 if len(val_data) % batch_size != 0 else 0)
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(val_data))
            batch_tensor = torch.tensor(val_data[start_idx:end_idx], dtype=torch.float32, device=flow_model.device)
            
            x0, x1 = batch_tensor[:, 0], batch_tensor[:, -1]
            batch_size_actual = x0.shape[0]
            t_values = torch.linspace(0.1, 0.9, 5).repeat(batch_size_actual, 1).to(flow_model.device)
            
            for t_idx in range(t_values.shape[1]):
                t = t_values[:, t_idx:t_idx+1]
                ut = x1 - x0
                xt = x0 + t * ut
                predicted_ut = flow_model.model(xt, t)
                loss = F.mse_loss(predicted_ut, ut)
                val_losses.append(loss.item())
    
    return sum(val_losses) / len(val_losses)

if __name__ == "__main__":
    """
    Example usage of the flow matching framework:
    1. Loads trajectory data
    2. Initializes and trains the model
    3. Creates visualizations
    4. Saves results and model checkpoints
    """
    # Load the NPZ file
    output_dir = 'flow_output'
    os.makedirs(output_dir, exist_ok=True)
    
    data = np.load('save_logs/2316385/s2ef_predictions.npz')
    trajectories = data['latents'].reshape(-1, 21, 128)
    
    print(f"Loaded trajectories shape: {trajectories.shape}", flush=True)
    
    # Train the residual flow matching model
    embedding_dim = 128  # Matches your latent dimension
    num_epochs = 1550
    batch_size = 32
    hidden_dims = [256, 512, 512, 256]
    
    # Train flow matching model
    flow_model, losses = train_flow_model(
        trajectories=trajectories,
        embedding_dim=embedding_dim,
        num_epochs=num_epochs,
        batch_size=batch_size,
        output_dir=output_dir,
        hidden_dims=hidden_dims
    )
    
    # Create comprehensive visualizations
    val_data = prepare_data(trajectories)[1]  # Get validation set
    flow_model.visualize(
        val_data[:20],
        step=val_data.shape[1]-1,
        output_dir=output_dir,
        plots=["2d", "3d", "tsne"]
    )
    
    print("Training completed and model saved!", flush=True)

