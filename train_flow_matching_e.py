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
from MPFlow.MPFlow import MPFlow
from MPFlow.EnergyPredictor import EnergyPredictor
    
    
class FlowMatching:
    """
    A class to manage flow matching on message passing trajectories.
    Implements training, sampling, and visualization of flow-based trajectories.
    
    Key features:
    - Supports multiple sampling methods (heun, rk4, dopri5)
    - Includes visualization tools (2D PCA, 3D PCA, t-SNE)
    - Implements cosine annealing with warm restarts
    
    Args:
        embedding_dim: Dimension of the embedding
        hidden_dims: List of hidden dimensions
        lr: Learning rate
        use_attention: Whether to use attention
        energy_checkpoint_path: Path to energy predictor checkpoint
        device: Device to run the model on
    """
    def __init__(
        self, 
        embedding_dim, 
        seq_length=49,
        hidden_dims=[256, 512, 512, 256], 
        lr=1e-4, 
        use_attention=True,
        energy_checkpoint_path=None,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.embedding_dim = embedding_dim
        
        # Initialize the residual flow matching network
        self.model = MPFlow(
            embedding_dim=embedding_dim,
            seq_length=seq_length,
            hidden_dims=hidden_dims,
            use_attention=use_attention
        ).to(device)
        
        # Initialize energy predictor if checkpoint path is provided
        self.energy_predictor = None
        if energy_checkpoint_path is not None:
            self.energy_predictor = EnergyPredictor(
                embedding_dim=embedding_dim,
                seq_length=seq_length,
                checkpoint_path=energy_checkpoint_path,
                device=device
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
    
    def sample_trajectory(self, x0, method="heun"):
        """
        Samples a trajectory using ultra-fast, high-accuracy ODE solver.
        
        Args:
            x0: Starting point
            method: Sampling method ("heun", "rk4", "dopri5")
        """
        self.model.eval()
        batch_size = x0.shape[0]
        device = x0.device
        
        x = x0.clone()
        with torch.no_grad():
            if method == "heun":
                # Previous Heun implementation
                t = torch.zeros((batch_size, 49, 1), device=device)
                k1 = self.model(x, t)
                x_pred = x + k1
                k2 = self.model(x_pred, t + 1.0)
                x = x + 0.5 * (k1 + k2)
            
            elif method == "rk4":
                # Original RK4 implementation
                t = torch.zeros((batch_size, 49, 1), device=device)
                k1 = self.model(x, t)
                k2 = self.model(x + 0.5 * k1, t + 0.5)
                k3 = self.model(x + 0.5 * k2, t + 0.5)
                k4 = self.model(x + k3, t + 1.0)
                x = x + (1.0 / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                
            elif method == "euler":
                # Simple Euler method
                t = torch.zeros((batch_size, 49, 1), device=device)
                velocity = self.model(x, t)
                x = x + velocity
            
        trajectory = torch.stack([x0, x], dim=1)
        return trajectory

    def predict_energy(self, x):
        """
        Predicts energy for given embeddings.
        
        Args:
            x: Input embeddings of shape (batch_size, seq_length, embedding_dim)
            
        Returns:
            torch.Tensor: Predicted energies of shape (batch_size,)
        """
        if self.energy_predictor is None:
            raise ValueError("Energy predictor not initialized. Provide energy_checkpoint_path during initialization.")
        
        with torch.no_grad():
            return self.energy_predictor(x)

    def sample_trajectory_with_energy(self, x0, method="heun"):
        """
        Samples a trajectory and computes energies along the path.
        
        Args:
            x0: Starting point
            method: Sampling method ("heun", "rk4", "dopri5")
            
        Returns:
            tuple: (trajectory, energies) where energies contains predictions for each point
        """
        trajectory = self.sample_trajectory(x0, method)
        
        if self.energy_predictor is not None:
            energies = []
            for t in range(trajectory.shape[1]):
                energy = self.predict_energy(trajectory[:, t])
                energies.append(energy)
            energies = torch.stack(energies, dim=1)  # Shape: (batch_size, timesteps)
            return trajectory, energies
        
        return trajectory, None

    def visualize(self, trajectories, step=100, output_dir='flow_output', plots=["2d", "3d"]):
        """
        Creates comprehensive visualizations of trajectories.
        Generates multiple plot types:
        - 2D PCA: Principal component projection
        - 3D PCA: 3D visualization of main components
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
        sampled_trajectories = self.sample_trajectory(x0, method="heun")
        
        # Calculate statistics
        rmse_per_sample = torch.sqrt(torch.mean((x1_gt.cpu() - sampled_trajectories[:, -1, :].cpu())**2, dim=1))
        avg_rmse = rmse_per_sample.mean().item()
        
        # Directory for this visualization run
        os.makedirs(output_dir, exist_ok=True)
        
        # Save statistics to file
        with open(os.path.join(output_dir, f"trajectory_stats_{step}.txt"), "w") as f:
            f.write(f"Average RMSE: {avg_rmse:.6f}\n")
            f.write(f"Per-sample RMSE: {rmse_per_sample.numpy()}\n")
        
        # Create visualizations
        if "2d" in plots:
            self._create_2d_pca_plot(ground_truth, sampled_trajectories, step, output_dir)
        
        if "3d" in plots:
            self._create_3d_pca_plot(ground_truth, sampled_trajectories, step, output_dir)
        
    def _create_2d_pca_plot(self, ground_truth, sampled_trajectories, step, output_dir):
        """
        Creates 2D PCA visualization comparing ground truth vs sampled trajectories.
        Handles 2D embeddings of shape (batch, time, 49, 128).
        """
        # Flatten sequence and embedding dimensions
        gt_flat = ground_truth.cpu().reshape(-1, ground_truth.shape[2] * ground_truth.shape[3])
        sampled_flat = sampled_trajectories.cpu().reshape(-1, sampled_trajectories.shape[2] * sampled_trajectories.shape[3])
        
        # Apply PCA on all points
        pca = PCA(n_components=2)
        gt_2d = pca.fit_transform(gt_flat)
        sampled_2d = pca.transform(sampled_flat)
        
        # Reshape back
        n_samples = ground_truth.shape[0]
        gt_timesteps = ground_truth.shape[1]
        sampled_timesteps = sampled_trajectories.shape[1]
        
        gt_2d = gt_2d.reshape(n_samples, gt_timesteps, 2)
        sampled_2d = sampled_2d.reshape(n_samples, sampled_timesteps, 2)
        
        # Create plot
        plt.figure(figsize=(12, 10))
        colors = plt.cm.tab20(np.linspace(0, 1, n_samples))
        
        for i in range(n_samples):
            # Plot trajectories and points
            plt.plot(gt_2d[i, :, 0], gt_2d[i, :, 1], '--', color=colors[i], alpha=0.7, linewidth=2)
            plt.plot(sampled_2d[i, :, 0], sampled_2d[i, :, 1], '-', color=colors[i], alpha=0.9, linewidth=2)
            
            # Mark start and end points
            plt.scatter(gt_2d[i, 0, 0], gt_2d[i, 0, 1], color='blue', s=50, marker='o')
            plt.scatter(gt_2d[i, -1, 0], gt_2d[i, -1, 1], color='red', s=50, marker='o')
            plt.scatter(sampled_2d[i, 0, 0], sampled_2d[i, 0, 1], color='blue', s=50, marker='o')
            plt.scatter(sampled_2d[i, -1, 0], sampled_2d[i, -1, 1], color='black', s=25, marker='x')
        
        plt.title('Flow Trajectories - PCA Projection\n'
                 f'(Explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}, '
                 f'PC2={pca.explained_variance_ratio_[1]:.1%})')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.grid(True, alpha=0.3)
        
        plt.legend([
            'Ground Truth', 
            'Sampled Trajectory',
            'Start Points',
            'End Points (GT)',
            'End Points (Sampled)'
        ], loc='upper right')
        
        plt.savefig(os.path.join(output_dir, f"flow_trajectories_2d_{step}.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def _create_3d_pca_plot(self, ground_truth, sampled_trajectories, step, output_dir):
        """
        Generates 3D PCA plots with interactive viewing capabilities.
        Handles 2D embeddings of shape (batch, time, 49, 128).
        """
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            # Flatten sequence and embedding dimensions
            gt_flat = ground_truth.cpu().reshape(-1, ground_truth.shape[2] * ground_truth.shape[3])
            sampled_flat = sampled_trajectories.cpu().reshape(-1, sampled_trajectories.shape[2] * sampled_trajectories.shape[3])
            
            # Apply PCA for 3D
            pca = PCA(n_components=3)
            gt_3d = pca.fit_transform(gt_flat)
            sampled_3d = pca.transform(sampled_flat)
            
            # Reshape back
            n_samples = ground_truth.shape[0]
            gt_timesteps = ground_truth.shape[1]
            sampled_timesteps = sampled_trajectories.shape[1]
            
            gt_3d = gt_3d.reshape(n_samples, gt_timesteps, 3)
            sampled_3d = sampled_3d.reshape(n_samples, sampled_timesteps, 3)
            
            # Create plot
            fig = plt.figure(figsize=(14, 12))
            ax = fig.add_subplot(111, projection='3d')
            colors = plt.cm.tab20(np.linspace(0, 1, n_samples))
            
            for i in range(n_samples):
                # Plot trajectories
                ax.plot(gt_3d[i, :, 0], gt_3d[i, :, 1], gt_3d[i, :, 2], '--', color=colors[i], alpha=0.7, linewidth=2)
                ax.plot(sampled_3d[i, :, 0], sampled_3d[i, :, 1], sampled_3d[i, :, 2], '-', color=colors[i], alpha=0.9, linewidth=2)
                
                # Mark points
                ax.scatter(gt_3d[i, 0, 0], gt_3d[i, 0, 1], gt_3d[i, 0, 2], color='blue', s=50, marker='o')
                ax.scatter(gt_3d[i, -1, 0], gt_3d[i, -1, 1], gt_3d[i, -1, 2], color='red', s=50, marker='o')
                ax.scatter(sampled_3d[i, 0, 0], sampled_3d[i, 0, 1], sampled_3d[i, 0, 2], color='blue', s=50, marker='o')
                ax.scatter(sampled_3d[i, -1, 0], sampled_3d[i, -1, 1], sampled_3d[i, -1, 2], color='black', s=25, marker='x')
            
            ax.set_title('Flow Trajectories - 3D PCA Projection\n'
                        f'(Explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}, '
                        f'PC2={pca.explained_variance_ratio_[1]:.1%}, '
                        f'PC3={pca.explained_variance_ratio_[2]:.1%})')
            ax.set_xlabel('First Principal Component')
            ax.set_ylabel('Second Principal Component')
            ax.set_zlabel('Third Principal Component')
            
            ax.legend([
                'Ground Truth', 
                'Sampled Trajectory',
                'Start Points',
                'End Points (GT)',
                'End Points (Sampled)'
            ])
            
            plt.savefig(os.path.join(output_dir, f"flow_trajectories_3d_{step}.png"), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error creating 3D plot: {e}", flush=True)

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
    hidden_dims=[256, 512, 512, 256],
    energy_checkpoint_path=None
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
        seq_length=49,
        hidden_dims=hidden_dims,
        lr=1e-4,
        use_attention=True,
        energy_checkpoint_path=energy_checkpoint_path,
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
            
            x0, x1 = batch_tensor[:, 0], batch_tensor[:, -1] # [batch_size, 49, 128]
            t = torch.rand(batch_size, 49, 1, device=flow_model.device) # [batch_size, 49, 1]
            
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
        
        if (epoch + 1) % 50 == 0 or epoch == num_epochs - 1 or epoch == 0:
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
            t_values = torch.linspace(0.1, 0.9, 5).repeat(batch_size_actual, 49, 1).to(flow_model.device)
            
            for t_idx in range(t_values.shape[2]):
                t = t_values[:, :, t_idx:t_idx+1]
                ut = x1 - x0
                xt = x0 + t * ut
                predicted_ut = flow_model.model(xt, t)
                loss = F.mse_loss(predicted_ut, ut)
                val_losses.append(loss.item())
    
    return sum(val_losses) / len(val_losses)


def load_trajectory_data():
    """
    Loads and combines trajectory data from multiple NPZ files.
    Properly shuffles the combined trajectories along the batch dimension.
    
    Returns:
        Combined and shuffled trajectories array of shape (n_trajectories, 2, 49, 128)
    """
    file_paths = [
        'save_logs/0/s2ef_predictions.npz',
        'save_logs/1/s2ef_predictions.npz', 
        'save_logs/2/s2ef_predictions.npz',
        'save_logs/3/s2ef_predictions.npz',
        'save_logs/4/s2ef_predictions.npz',
        'save_logs/5/s2ef_predictions.npz',
        # 'save_logs/6/s2ef_predictions.npz',
        # 'save_logs/7/s2ef_predictions.npz',
    ]
        
    all_trajectories = []
    for file_path in file_paths:
        data = np.load(file_path)
        trajectories = data['latents'].reshape(-1, 2, 49, 128)
        all_trajectories.append(trajectories)
    
    # First concatenate all trajectories
    combined_trajectories = np.concatenate(all_trajectories, axis=0)
    
    # Then set random seed and shuffle along batch dimension
    np.random.seed(42)
    shuffle_indices = np.random.permutation(len(combined_trajectories))
    combined_trajectories = combined_trajectories[shuffle_indices]
    
    return combined_trajectories
    

if __name__ == "__main__":
    """
    Example usage of the flow matching framework:
    1. Loads trajectory data
    2. Initializes and trains the model
    3. Creates visualizations
    4. Saves results and model checkpoints
    """
    # Load the NPZ file
    output_dir = 'flow_output/exp1'
    os.makedirs(output_dir, exist_ok=True)
    
    trajectories = load_trajectory_data()
    print(f"Loaded trajectories shape: {trajectories.shape}", flush=True)
    
    # Train the residual flow matching model
    embedding_dim = 128
    num_epochs = 1550
    batch_size = 64
    hidden_dims = [256, 512, 512, 256]  
    energy_checkpoint_path = 'models/oc20/s2ef/2M/equiformer_v2/N@12_L@6_M@2/bs@64_lr@2e-4_wd@1e-3_epochs@12_warmup-epochs@0.1_g@1x1/eq2_153M_ec4_allmd.pt'
    
    # Train flow matching model
    flow_model, losses = train_flow_model(
        trajectories=trajectories,
        embedding_dim=embedding_dim,
        num_epochs=num_epochs,
        batch_size=batch_size,
        output_dir=output_dir,
        hidden_dims=hidden_dims,
        energy_checkpoint_path=energy_checkpoint_path
    )
    
    # Create comprehensive visualizations
    val_data = prepare_data(trajectories)[1]  # Get validation set
    flow_model.visualize(
        val_data[:20],
        step=val_data.shape[1]-1,
        output_dir=output_dir,
        plots=["2d", "3d"]
    )
    
    print("Training completed and model saved!", flush=True)

