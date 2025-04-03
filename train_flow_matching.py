import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
import logging
from sklearn.decomposition import PCA
from torch.optim.lr_scheduler import OneCycleLR
from nets.equiformer_v2.MPFlow import EquivariantMPFlow
from nets.equiformer_v2.so3 import SO3_Grid
from nets.equiformer_v2.module_list import ModuleListInfo
import glob
    
    
class FlowMatching:
    """
    A class to manage flow matching on message passing trajectories.
    Implements training, sampling, and visualization of flow-based trajectories.
    
    Key features:
    - Supports multiple sampling methods (heun, rk4, dopri5)
    - Includes visualization tools (2D PCA, 3D PCA, t-SNE)
    - Implements cosine annealing with warm restarts
    
    Args:
        lr: Learning rate
        device: Device to run the model on
    """
    def __init__(
        self, 
        lr=1e-4, 
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.sphere_channels = 128
        self.ffn_hidden_channels = 256
        self.time_embed_dim = 128
        self.lmax_list = [6]
        self.mmax_list = [3]
        self.grid_resolution = 18
        self.ffn_activation = 'silu'
        self.norm_type = 'layer_norm_sh'
        self.use_gate_act = False
        self.use_grid_mlp = True
        self.use_sep_s2_act = True
        self.num_layers = 4
        self.proj_drop = 0.0
        
        # Initialize the residual flow matching network
        self.SO3_grid = ModuleListInfo('({}, {})'.format(max(self.lmax_list), max(self.lmax_list)))
        for l in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        l, 
                        m, 
                        resolution=self.grid_resolution, 
                        normalization='component'
                    )
                )
            self.SO3_grid.append(SO3_m_grid)
            
        self.model = EquivariantMPFlow(
            sphere_channels=self.sphere_channels,
            ffn_hidden_channels=self.ffn_hidden_channels,
            time_embed_dim=self.time_embed_dim,
            lmax_list=self.lmax_list,
            mmax_list=self.mmax_list, # Pass mmax_list
            SO3_grid=self.SO3_grid,     # Pass SO3_grid
            activation=self.ffn_activation, # Reuse ffn activation
            norm_type=self.norm_type,      # Reuse norm type
            use_gate_act=self.use_gate_act,
            use_grid_mlp=self.use_grid_mlp,
            use_sep_s2_act=self.use_sep_s2_act,
            num_layers=self.num_layers,
            proj_drop=self.proj_drop, # Pass if needed and supported by EquivariantMPFlow's blocks
        )
        self.model.to(self.device)
        
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
        # Keep original t with shape (B, 1) for the model
        # Create a broadcastable version for xt calculation
        t_broadcast = t.view(batch_size, 1, 1)
        
        ut = x1 - x0
        xt = x0 + t_broadcast * ut # Use broadcastable t here
            
        # Get the model's prediction
        predicted_ut = self.model(xt, t) # Use original t (B, 1) here
        
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
                # Time tensor should be [batch_size, 1]
                t = torch.zeros((batch_size, 1), device=device)
                k1 = self.model(x, t)
                x_pred = x + k1
                # Time t+1 should also be [batch_size, 1]
                t_next = torch.ones((batch_size, 1), device=device)
                k2 = self.model(x_pred, t_next)
                x = x + 0.5 * (k1 + k2)
            
            elif method == "rk4":
                # Time tensors should be [batch_size, 1]
                t = torch.zeros((batch_size, 1), device=device)
                t_half = torch.full((batch_size, 1), 0.5, device=device)
                t_one = torch.ones((batch_size, 1), device=device)
                k1 = self.model(x, t)
                k2 = self.model(x + 0.5 * k1, t_half)
                k3 = self.model(x + 0.5 * k2, t_half)
                k4 = self.model(x + k3, t_one)
                x = x + (1.0 / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                
            elif method == "euler":
                # Time tensor should be [batch_size, 1]
                t = torch.zeros((batch_size, 1), device=device)
                velocity = self.model(x, t)
                x = x + velocity
            
        trajectory = torch.stack([x0, x], dim=1)
        return trajectory

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
        logger = logging.getLogger() # Get logger instance
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
            logger.error(f"Error creating 3D plot: {e}") # Use logger.error

    def save_model(self, path):
        """Save the model and optimizer state."""
        logger = logging.getLogger()
        logger.info(f"Saving model checkpoint to {path}")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path):
        """Load the model and optimizer state."""
        logger = logging.getLogger()
        logger.info(f"Loading model checkpoint from {path}")
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
    num_epochs=1000,
    batch_size=32,
    output_dir='flow_output',
    validation_interval=10,
    lr=1e-4,
    logger=None # Add logger parameter
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
    if logger is None: # Basic fallback if no logger is provided
        logger = logging.getLogger()

    os.makedirs(output_dir, exist_ok=True)
    logger.info("Preparing data...")
    train_data, val_data = prepare_data(trajectories, train_ratio=0.8)
    logger.info(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")

    logger.info("Initializing FlowMatching model...")
    flow_model = FlowMatching(lr=lr)
    logger.info(f"Model parameters: {sum(p.numel() for p in flow_model.model.parameters())}")

    model_path = os.path.join(output_dir, 'flow_matching_model.pt')
    if os.path.exists(model_path):
        try:
            flow_model.load_model(model_path)
            logger.info("Loaded existing model for continued training.")
        except Exception as e:
            logger.warning(f"Could not load existing model from {model_path}, starting fresh. Error: {e}")

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience, patience_counter = 100, 0
    logger.info(f"Starting training for {num_epochs} epochs with batch size {batch_size}.")

    # Use logger's stream handler for tqdm compatibility
    tqdm_stream = None
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
             #tqdm_stream = handler.stream # Not strictly needed as tqdm defaults to stderr
             pass # Keep default tqdm behavior (stderr) unless explicit stdout needed

    for epoch in tqdm(range(num_epochs), file=tqdm_stream, desc="Training Progress"):
        flow_model.model.train() # Ensure model is in training mode
        epoch_losses = []
        num_batches = len(train_data) // batch_size

        for i in range(num_batches):
            batch_indices = np.random.choice(len(train_data), batch_size, replace=False)
            batch_tensor = torch.tensor(train_data[batch_indices], dtype=torch.float32, device=flow_model.device)

            x0, x1 = batch_tensor[:, 0], batch_tensor[:, -1] # [batch_size, 49, 128]
            t = torch.rand(batch_size, 1, device=flow_model.device) # Sample time per batch item

            loss = flow_model.train_step(x0, x1, t)
            epoch_losses.append(loss)

        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        train_losses.append(avg_loss)

        # Log training loss less frequently to avoid excessive output
        if (epoch + 1) % validation_interval == 0 or epoch == num_epochs - 1:
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, Average Training Loss: {avg_loss:.6f}")

        if (epoch + 1) % validation_interval == 0 or epoch == num_epochs - 1:
            logger.info(f"--- Starting Validation for Epoch {epoch + 1} ---")
            val_loss = validate_model(flow_model, val_data, batch_size, logger=logger) # Pass logger
            val_losses.append(val_loss)
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.6f}")

            if val_loss < best_val_loss:
                logger.info(f"New best validation loss: {val_loss:.6f} (previous: {best_val_loss:.6f}). Saving model.")
                best_val_loss = val_loss
                flow_model.save_model(os.path.join(output_dir, 'best_flow_model.pt'))
                patience_counter = 0
            else:
                patience_counter += 1
                logger.info(f"Validation loss did not improve. Patience counter: {patience_counter}/{patience}")

            if patience_counter >= patience:
                logger.warning(f"Early stopping triggered after {epoch + 1} epochs due to lack of validation improvement.")
                break

        flow_model.scheduler.step()

        if (epoch + 1) % 100 == 0 or epoch == num_epochs - 1 or epoch == 0: # Visualize less frequently
             logger.info(f"--- Generating Visualization for Epoch {epoch + 1} ---")
             flow_model.visualize(
                 val_data[:50], # Visualize on a subset of validation data
                 step=epoch + 1, # Log the epoch number
                 output_dir=output_dir,
                 plots=["2d", "3d"]
             )
             logger.info(f"--- Visualization for Epoch {epoch + 1} saved ---")

    logger.info(f"Saving final model state to {model_path}")
    flow_model.save_model(model_path)

    # Plot training loss
    logger.info("Plotting training loss curve.")
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, 'b-', linewidth=2)
    plt.xlabel('Epoch'), plt.ylabel('Loss')
    plt.title('Training Loss Progress')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'training_loss_curve.png'))
    plt.close()

    # Plot validation loss
    logger.info("Plotting validation loss curve.")
    plt.figure(figsize=(12, 6))
    val_epochs = list(range(validation_interval -1, epoch + 1, validation_interval)) # Adjust epoch indices
    if len(val_losses) != len(val_epochs): # Handle early stopping case
         val_epochs = list(range(validation_interval -1, epoch - (epoch % validation_interval) + 1, validation_interval))

    # Ensure lengths match before plotting
    plot_len = min(len(val_epochs), len(val_losses))
    plt.plot(val_epochs[:plot_len], val_losses[:plot_len], 'r-', linewidth=2)
    plt.xlabel('Epoch'), plt.ylabel('Loss')
    plt.title('Validation Loss Progress')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'validation_loss_curve.png'))
    plt.close()

    logger.info("Loading best model based on validation loss.")
    flow_model.load_model(os.path.join(output_dir, 'best_flow_model.pt'))
    return flow_model, {'train': train_losses, 'val': val_losses}

def validate_model(flow_model, val_data, batch_size=32, logger=None): # Add logger parameter
    """
    Evaluates model performance on validation data.
    Uses multiple time points for robust evaluation.
    Returns average loss across all validation samples.
    """
    if logger is None:
        logger = logging.getLogger()

    logger.info("Starting validation...")
    flow_model.model.eval() # Set model to evaluation mode
    val_losses = []

    with torch.no_grad():
        num_batches = len(val_data) // batch_size + (1 if len(val_data) % batch_size != 0 else 0)
        logger.info(f"Processing {len(val_data)} validation samples in {num_batches} batches.")

        for i in tqdm(range(num_batches), desc="Validation Progress", leave=False): # Add progress bar
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(val_data))
            if start_idx >= end_idx: continue # Skip empty batch

            batch_tensor = torch.tensor(val_data[start_idx:end_idx], dtype=torch.float32, device=flow_model.device)

            x0, x1 = batch_tensor[:, 0], batch_tensor[:, -1]
            batch_size_actual = x0.shape[0]

            # Consistent t sampling for validation is better
            # Sample a single set of times and broadcast
            t = torch.rand(batch_size_actual, 1, 1, device=flow_model.device).expand(-1, x0.shape[1], -1) # (B, 49, 1)

            ut = x1 - x0 # Shape (B, 49, D)
            xt = x0 + t * ut # Broadcasting: (B,49,D) + (B,49,1)*(B,49,D) -> (B,49,D)

            # Model expects t with shape (B, 1)
            t_model = t[:, 0, :] # Take time from first seq element -> Shape (B, 1)
            predicted_ut = flow_model.model(xt, t_model)
            loss = F.mse_loss(predicted_ut, ut) # Calculate loss over the full output
            val_losses.append(loss.item())

    avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0
    logger.info(f"Validation finished. Average loss: {avg_val_loss:.6f}")
    return avg_val_loss


def load_trajectory_data():
    """
    Loads and combines trajectory data from multiple NPZ files more memory-efficiently.
    Pre-allocates the final array and loads data directly into slices.
    Shuffles the combined trajectories along the batch dimension after loading.

    Returns:
        Combined and shuffled trajectories array of shape (n_trajectories, 2, 49, 128)
    """
    logger = logging.getLogger() # Get logger
    data_dir = 'mpflow_data'
    file_pattern = os.path.join(data_dir, 'embeddings_*.npz')
    file_paths = sorted(glob.glob(file_pattern))
    logger.info(f"Searching for trajectory files in {data_dir} matching pattern 'embeddings_*.npz'. Found {len(file_paths)} files.")


    if not file_paths:
        logger.error(f"No NPZ files found in {data_dir} matching pattern.")
        raise ValueError(f"No NPZ files found in {data_dir}/")

    # First pass: determine total number of samples and data shape
    total_samples = 0
    sample_shape = None
    dtype = None
    logger.info("Scanning files to determine total size and shape...")
    valid_files = [] # Keep track of files that pass checks
    for i, file_path in enumerate(file_paths):
        try:
            with np.load(file_path) as data:
                # Check if required keys exist
                if 'x0' not in data or 'x1' not in data:
                     logger.warning(f"Skipping {file_path}: missing 'x0' or 'x1' key.")
                     continue

                x0_shape = data['x0'].shape
                x1_shape = data['x1'].shape

                if x0_shape[0] != x1_shape[0]:
                    logger.warning(f"Skipping {file_path}: mismatched sample count between x0 ({x0_shape[0]}) and x1 ({x1_shape[0]}).")
                    continue

                current_samples = x0_shape[0]
                if current_samples == 0:
                    logger.warning(f"Skipping {file_path}: contains 0 samples.")
                    continue

                # Check shape consistency
                expected_inner_shape = (49, 128)
                if len(x0_shape) != 3 or x0_shape[1:] != expected_inner_shape:
                     logger.warning(f"Skipping {file_path}: unexpected x0 shape {x0_shape}. Expected (N, {expected_inner_shape[0]}, {expected_inner_shape[1]}).")
                     continue
                if len(x1_shape) != 3 or x1_shape[1:] != expected_inner_shape:
                     logger.warning(f"Skipping {file_path}: unexpected x1 shape {x1_shape}. Expected (N, {expected_inner_shape[0]}, {expected_inner_shape[1]}).")
                     continue

                # Store shape and dtype from the first valid file
                current_data_shape = (2,) + x0_shape[1:] # (2, 49, 128)
                current_dtype = data['x0'].dtype

                if sample_shape is None:
                    sample_shape = current_data_shape
                    dtype = current_dtype
                    logger.info(f"Determined sample shape: {sample_shape}, dtype: {dtype} from {file_path}")
                elif current_data_shape != sample_shape:
                     logger.warning(f"Skipping {file_path}: shape {current_data_shape} mismatches expected {sample_shape}.")
                     continue
                elif current_dtype != dtype:
                     logger.warning(f"Skipping {file_path}: dtype {current_dtype} mismatches expected {dtype}.")
                     continue

                # If all checks pass, count samples and add to valid list
                total_samples += current_samples
                valid_files.append(file_path)


            if (i + 1) % 50 == 0: # Log progress occasionally
                 logger.info(f"Scanned {i+1}/{len(file_paths)} files. Current valid samples: {total_samples}")

        except Exception as e:
            logger.error(f"Error reading metadata from {file_path}: {e}. Skipping.")
            continue # Skip file on error

    if total_samples == 0 or sample_shape is None:
        logger.error("No valid trajectory data could be processed from any files.")
        raise ValueError("No valid trajectory data found or could be processed.")

    logger.info(f"Scan complete. Total valid samples: {total_samples}, Sample shape: {sample_shape}, Dtype: {dtype}")

    # Pre-allocate the array
    final_shape = (total_samples,) + sample_shape
    logger.info(f"Allocating array with shape: {final_shape}")
    try:
        combined_trajectories = np.zeros(final_shape, dtype=dtype)
    except MemoryError:
         logger.error(f"MemoryError: Failed to allocate array of shape {final_shape} and dtype {dtype}. Insufficient RAM.")
         raise
    logger.info("Array allocation successful.")

    # Second pass: load data into the pre-allocated array using only valid files
    current_pos = 0
    logger.info(f"Loading data from {len(valid_files)} valid files into pre-allocated array...")
    for i, file_path in enumerate(tqdm(valid_files, desc="Loading Files")): # Use tqdm for progress
         try:
             with np.load(file_path) as data:
                 # No need for extensive checks here, already done in first pass
                 x0 = data['x0'] # [N, 49, 128]
                 x1 = data['x1'] # [N, 49, 128]
                 num_samples_in_file = x0.shape[0]

                 if current_pos + num_samples_in_file > total_samples:
                      logger.warning(f"Data from {file_path} exceeds allocated space. This shouldn't happen. Truncating load.")
                      num_samples_in_file = total_samples - current_pos # Load only what fits
                      if num_samples_in_file <= 0: continue # Skip if no space left

                 # Stack and place into the correct slice
                 combined_trajectories[current_pos : current_pos + num_samples_in_file, 0] = x0[:num_samples_in_file]
                 combined_trajectories[current_pos : current_pos + num_samples_in_file, 1] = x1[:num_samples_in_file]

                 current_pos += num_samples_in_file

                 # Log less frequently during loading
                 # if (i + 1) % 10 == 0 or i == len(valid_files) - 1:
                 #     logger.debug(f"Loaded {i+1}/{len(valid_files)}: {file_path} ({num_samples_in_file} samples). Current position: {current_pos}")


         except Exception as e:
             logger.error(f"Error loading data from {file_path} during second pass: {e}. Skipping file, data may be incomplete.")
             # Don't increment current_pos if load fails

    # Ensure we filled the array as expected
    if current_pos != total_samples:
        logger.warning(f"Expected to load {total_samples} samples, but loaded {current_pos}. Trimming array.")
        combined_trajectories = combined_trajectories[:current_pos]


    # Shuffle along the batch dimension (axis 0)
    logger.info(f"Shuffling {len(combined_trajectories)} trajectories...")
    np.random.seed(42) # for reproducibility
    shuffle_indices = np.random.permutation(len(combined_trajectories))
    combined_trajectories = combined_trajectories[shuffle_indices]
    logger.info("Data loading and shuffling complete.")

    return combined_trajectories


if __name__ == "__main__":
    """
    Example usage of the flow matching framework:
    1. Loads trajectory data
    2. Initializes and trains the model
    3. Creates visualizations
    4. Saves results and model checkpoints
    """
    # --- Basic Setup ---
    output_dir = 'flow_output/exp1'
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'training.out')

    # --- Configure Logging ---
    logging.basicConfig(
        level=logging.INFO, # Set base level
        format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s', # Added function name
        datefmt='%Y-%m-%d %H:%M:%S', # Added date format
        handlers=[
            logging.FileHandler(log_file, mode='w'), # Overwrite log file each run
            logging.StreamHandler(sys.stdout) # Log to console
        ]
    )
    logger = logging.getLogger()
    # Example: Set different level for a specific module if needed
    # logging.getLogger('matplotlib').setLevel(logging.WARNING)

    logger.info("--- Starting Training Script ---")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Torch device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")


    # --- Load Data ---
    logger.info("Loading trajectory data...")
    try:
        trajectories = load_trajectory_data()
        logger.info(f"Loaded trajectories shape: {trajectories.shape}")
    except (ValueError, MemoryError) as e:
         logger.error(f"Failed to load data: {e}")
         sys.exit(1) # Exit if data loading fails
    except Exception as e:
         logger.exception("An unexpected error occurred during data loading.") # Log full traceback
         sys.exit(1)


    # --- Train Model ---
    logger.info("Starting model training...")
    num_epochs = 1550
    batch_size = 128
    learning_rate = 1e-4

    try:
        flow_model, losses = train_flow_model(
            trajectories=trajectories,
            num_epochs=num_epochs,
            batch_size=batch_size,
            output_dir=output_dir,
            lr=learning_rate,
            logger=logger # Pass the configured logger
        )
        logger.info("Model training finished.")
    except Exception as e:
         logger.exception("An unexpected error occurred during model training.") # Log full traceback
         sys.exit(1)


    # --- Visualize Results ---
    logger.info("Generating final visualizations using the best model...")
    try:
        train_data, val_data = prepare_data(trajectories) # Split again to get val_data easily
        flow_model.visualize(
            val_data[:20], # Visualize a small subset
            step='final', # Indicate final visualization
            output_dir=output_dir,
            plots=["2d", "3d"]
        )
        logger.info("Final visualizations saved.")
    except Exception as e:
        logger.exception("An unexpected error occurred during visualization.")


    logger.info("--- Training script completed successfully! ---")