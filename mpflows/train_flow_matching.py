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
from torch.optim.lr_scheduler import ExponentialLR
from nets.equiformer_v2.MPFlow import EquivariantMPFlow
from nets.equiformer_v2.so3 import SO3_Grid
from nets.equiformer_v2.module_list import ModuleListInfo
import glob
from typing import Tuple, Dict, List, Optional

# --- Constants ---
# Data Shape
TIME_DIM = 2
NODE_DIM = 49 # Example, adjust if necessary based on data
FEATURE_DIM = 128 # Example, adjust if necessary based on data
EMBEDDING_SHAPE = (NODE_DIM, FEATURE_DIM)
TRAJECTORY_SHAPE = (TIME_DIM,) + EMBEDDING_SHAPE # (2, 49, 128)

# Model Hyperparameters
SPHERE_CHANNELS = 128
FFN_HIDDEN_CHANNELS = 256
TIME_EMBED_DIM = 128
LMAX_LIST = [6]
MMAX_LIST = [3]
GRID_RESOLUTION = 18
FFN_ACTIVATION = 'silu'
NORM_TYPE = 'layer_norm_sh'
USE_GATE_ACT = False
USE_GRID_MLP = True
USE_SEP_S2_ACT = True
NUM_LAYERS = 6
PROJ_DROP = 0.0

# Training Hyperparameters
DEFAULT_LR = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-3
DEFAULT_BETAS = (0.9, 0.999)
DEFAULT_PATIENCE = 100
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_BATCH_SIZE = 128
DEFAULT_NUM_EPOCHS = 300
VALIDATION_INTERVAL = 10
VISUALIZATION_INTERVAL = 50
TIME_SAMPLING_EPS = 1e-5

# Visualization
PCA_N_COMPONENTS_2D = 2
PCA_N_COMPONENTS_3D = 3
MAX_VIZ_SAMPLES = 8
PLOT_DPI = 300

# File Paths & Names
DEFAULT_OUTPUT_DIR = f'flow_output/exp1_lr_{DEFAULT_LR}'
BEST_MODEL_FILENAME = 'best_flow_model.pt'
FINAL_MODEL_FILENAME = 'flow_matching_model.pt'
TRAINING_LOG_FILENAME = 'training.out'
TRAJ_STATS_FILENAME_TEMPLATE = "trajectory_stats_step{}.txt"
PLOT_2D_FILENAME_TEMPLATE = "flow_trajectories_2d_step{}.png"
PLOT_3D_FILENAME_TEMPLATE = "flow_trajectories_3d_step{}.png"
TRAIN_LOSS_PLOT_FILENAME = 'training_loss_curve.png'
VAL_LOSS_PLOT_FILENAME = 'validation_loss_curve.png'
DEFAULT_DATA_DIR = 'mpflow_data'
DATA_FILE_PATTERN = 'embeddings_*.npz'

logger = logging.getLogger(__name__)
    
    
class FlowMatching:
    """
    Manages Conditional Flow Matching model training and trajectory sampling.

    Handles model initialization, training steps, trajectory generation using
    ODE solvers, and visualization of results.
    """
    def __init__(
        self, 
        lr: float = DEFAULT_LR,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        betas: Tuple[float, float] = DEFAULT_BETAS,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = torch.device(device)
        logger.info(f"Initializing FlowMatching model on device: {self.device}")

        # --- Model Configuration ---
        self.sphere_channels = SPHERE_CHANNELS
        self.ffn_hidden_channels = FFN_HIDDEN_CHANNELS
        self.time_embed_dim = TIME_EMBED_DIM
        self.lmax_list = LMAX_LIST
        self.mmax_list = MMAX_LIST
        self.grid_resolution = GRID_RESOLUTION
        self.ffn_activation = FFN_ACTIVATION
        self.norm_type = NORM_TYPE
        self.use_gate_act = USE_GATE_ACT
        self.use_grid_mlp = USE_GRID_MLP
        self.use_sep_s2_act = USE_SEP_S2_ACT
        self.num_layers = NUM_LAYERS
        self.proj_drop = PROJ_DROP
        # --- End Model Configuration ---

        self.SO3_grid = self._build_so3_grid()
        self.model = self._build_model()
        self.model.to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay
        )
        logger.info(f"Optimizer: AdamW (lr={lr}, weight_decay={weight_decay})")

    def _build_so3_grid(self) -> ModuleListInfo:
        """Builds the SO3 grid structure."""
        max_l = max(self.lmax_list)
        so3_grid = ModuleListInfo(f'({max_l}, {max_l})')
        for l in range(max_l + 1):
            so3_m_grid = nn.ModuleList()
            # Note: Original code used max(lmax_list) for m range, check if intended
            for m in range(max_l + 1):
                so3_m_grid.append(
                    SO3_Grid(
                        l, 
                        m, 
                        resolution=self.grid_resolution, 
                        normalization='component'
                    )
                )
            so3_grid.append(so3_m_grid)
        return so3_grid
            
    def _build_model(self) -> EquivariantMPFlow:
        """Builds the EquivariantMPFlow model."""
        model = EquivariantMPFlow(
            sphere_channels=self.sphere_channels,
            ffn_hidden_channels=self.ffn_hidden_channels,
            time_embed_dim=self.time_embed_dim,
            lmax_list=self.lmax_list,
            mmax_list=self.mmax_list,
            SO3_grid=self.SO3_grid,
            activation=self.ffn_activation,
            norm_type=self.norm_type,
            use_gate_act=self.use_gate_act,
            use_grid_mlp=self.use_grid_mlp,
            use_sep_s2_act=self.use_sep_s2_act,
            num_layers=self.num_layers,
            proj_drop=self.proj_drop,
        )
        logger.info(f"EquivariantMPFlow model created with {sum(p.numel() for p in model.parameters()):,} parameters.")
        return model

    def train_step(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> float:
        """
        Performs a single training step using the Conditional Flow Matching loss.

        Args:
            x0: Input tensor at time 0 [batch, N, D].
            x1: Input tensor at time 1 [batch, N, D].
            t: Time tensor [batch, 1].

        Returns:
            The calculated MSE loss for the step.
        """
        batch_size = x0.shape[0]
        # Ensure t has the correct shape for broadcasting: [batch, 1, 1]
        t_broadcast = t.view(batch_size, 1, 1)
        
        # Conditional flow target: u_t = x1 - x0
        # Conditional flow sample: x_t = x0 + t * u_t
        ut_target = x1 - x0
        xt_sample = x0 + t_broadcast * ut_target

        # Predict the velocity ut_target given xt_sample and t
        ut_predicted = self.model(xt_sample, t)

        # Flow Matching Loss (MSE between predicted and target velocity)
        loss = F.mse_loss(ut_predicted, ut_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: Gradient clipping can be added here if needed
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def sample_trajectory(self, x0: torch.Tensor, method: str = "euler", n_steps: int = 1) -> torch.Tensor:
        """
        Generates a trajectory from x0 to x1 using a specified ODE solver method.

        Note: For 'heun' and 'rk4', this implementation performs a single step
              prediction from t=0 to t=1, suitable for flow matching where the
              velocity field directly maps the path. For 'euler', it performs
              `n_steps` integration steps.
        
        Args:
            x0: Starting point tensor [batch, N, D].
            method: Sampling method ("heun", "rk4", "euler").
            n_steps: Number of integration steps (used only for "euler").

        Returns:
            A tensor containing the trajectory [batch, 2, N, D], where the
            first time step is x0 and the second is the predicted x1.
            For Euler with n_steps > 1, intermediate steps are not returned.
        """
        self.model.eval() # Ensure model is in evaluation mode
        batch_size = x0.shape[0]
        device = x0.device
        
        x_current = x0.clone()

        with torch.no_grad():
            if method == "heun":
                # Single-step Heun's method (Predictor-Corrector) from t=0 to t=1
                t0 = torch.zeros((batch_size, 1), device=device)
                t1 = torch.ones((batch_size, 1), device=device)

                v0 = self.model(x_current, t0)          # Velocity at t=0
                x_pred_euler = x_current + v0           # Euler predictor step (h=1)
                v1_pred = self.model(x_pred_euler, t1)  # Velocity at predicted x1
                x_final = x_current + 0.5 * (v0 + v1_pred) # Corrector step (h=1)
            
            elif method == "rk4":
                # Single-step Runge-Kutta 4th order from t=0 to t=1
                t0 = torch.zeros((batch_size, 1), device=device)
                t_half = torch.full((batch_size, 1), 0.5, device=device)
                t1 = torch.ones((batch_size, 1), device=device)

                k1 = self.model(x_current, t0)
                k2 = self.model(x_current + 0.5 * k1, t_half) # Step h=1 assumed
                k3 = self.model(x_current + 0.5 * k2, t_half) # Step h=1 assumed
                k4 = self.model(x_current + k3, t1)       # Step h=1 assumed
                x_final = x_current + (1.0 / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                
            elif method == "euler":
                # Multi-step Forward Euler method
                dt = 1.0 / n_steps
                x_t = x_current # Start at x0
                for i in range(n_steps):
                    t_current = torch.full((batch_size, 1), i * dt, device=device)
                    velocity = self.model(x_t, t_current)
                    x_t = x_t + dt * velocity
                x_final = x_t # Final state after n_steps

            else:
                logger.error(f"Unsupported sampling method: {method}")
                raise ValueError(f"Unsupported sampling method: {method}")

        # Stack initial state (x0) and final predicted state (x1_pred)
        trajectory = torch.stack([x0, x_final], dim=1) # Shape [batch, 2, N, D]
        return trajectory

    def visualize(
        self,
        ground_truth_trajectories: np.ndarray,
        step: int,
        output_dir: str,
        plots: List[str] = ["2d", "3d"],
        sampling_method: str = "euler",
        sampling_steps: int = 1
    ):
        """
        Creates visualizations comparing ground truth and sampled trajectories.

        Generates 2D/3D PCA plots and saves trajectory statistics (RMSE).

        Args:
            ground_truth_trajectories: NumPy array of ground truth trajectories
                                       [num_samples, 2, N, D].
            step: Current training step/epoch number for filenames.
            output_dir: Directory to save visualizations and stats.
            plots: List of plot types to generate ("2d", "3d").
            sampling_method: Method used for generating sampled trajectories.
            sampling_steps: Number of steps for the sampling method.
        """
        logger.info(f"Generating visualizations for step {step}...")
        os.makedirs(output_dir, exist_ok=True)

        # Convert ground truth to tensor and select subset for visualization
        gt_tensor = torch.tensor(ground_truth_trajectories, dtype=torch.float32, device=self.device)
        num_viz_samples = min(MAX_VIZ_SAMPLES, gt_tensor.shape[0])
        if num_viz_samples == 0:
            logger.warning("No samples available for visualization.")
            return
        gt_subset = gt_tensor[:num_viz_samples] # Shape [viz_batch, 2, N, D]

        # Extract start (x0) and end (x1_gt) points
        x0 = gt_subset[:, 0]      # Shape [viz_batch, N, D]
        x1_gt = gt_subset[:, -1]  # Shape [viz_batch, N, D]

        # Sample trajectories starting from x0
        # sample_trajectory returns [viz_batch, 2, N, D]
        sampled_trajectories = self.sample_trajectory(x0, method=sampling_method, n_steps=sampling_steps)
        x1_sampled = sampled_trajectories[:, -1] # Shape [viz_batch, N, D]

        # Calculate Root Mean Squared Error (RMSE) between gt and sampled end points
        rmse_per_sample = torch.sqrt(torch.mean((x1_gt - x1_sampled)**2, dim=(1, 2))) # Avg over N, D
        avg_rmse = rmse_per_sample.mean().item()
        logger.info(f"Visualization Step {step}: Average RMSE = {avg_rmse:.6f}")

        # Save statistics
        stats_filename = TRAJ_STATS_FILENAME_TEMPLATE.format(step)
        stats_path = os.path.join(output_dir, stats_filename)
        try:
            with open(stats_path, "w") as f:
                f.write(f"Visualization Step: {step}\n")
                f.write(f"Sampling Method: {sampling_method} ({sampling_steps} steps)\n")
                f.write(f"Number of Samples Visualized: {num_viz_samples}\n")
                f.write(f"Average RMSE: {avg_rmse:.6f}\n")
                f.write("Per-sample RMSE:\n")
                for i, rmse in enumerate(rmse_per_sample.cpu().numpy()):
                    f.write(f"  Sample {i}: {rmse:.6f}\n")
            logger.info(f"Trajectory statistics saved to {stats_path}")
        except IOError as e:
            logger.error(f"Failed to write trajectory stats to {stats_path}: {e}")


        # Generate plots
        if "2d" in plots:
            self._create_pca_plot(gt_subset, sampled_trajectories, PCA_N_COMPONENTS_2D, step, output_dir)
        if "3d" in plots:
            self._create_pca_plot(gt_subset, sampled_trajectories, PCA_N_COMPONENTS_3D, step, output_dir)
        logger.info(f"Visualizations for step {step} completed.")

    def _create_pca_plot(
        self,
        ground_truth: torch.Tensor,
        sampled: torch.Tensor,
        n_components: int,
        step: int,
        output_dir: str
    ):
        """
        Creates a 2D or 3D PCA visualization comparing trajectories.

        Args:
            ground_truth: Ground truth trajectories tensor [batch, 2, N, D].
            sampled: Sampled trajectories tensor [batch, 2, N, D].
            n_components: Number of PCA components (2 or 3).
            step: Current training step/epoch number.
            output_dir: Directory to save the plot.
        """
        if n_components not in [PCA_N_COMPONENTS_2D, PCA_N_COMPONENTS_3D]:
            raise ValueError("n_components must be 2 or 3 for PCA plot.")
        if n_components == 3:
            try:
                from mpl_toolkits.mplot3d import Axes3D # Import only when needed
            except ImportError:
                logger.warning("mpl_toolkits.mplot3d not found. Skipping 3D PCA plot.")
                return

        n_samples = ground_truth.shape[0]
        n_timesteps = ground_truth.shape[1] # Should be 2 (start and end)
        embedding_dim = ground_truth.shape[2] * ground_truth.shape[3] # N * D

        # Flatten time and embedding dimensions for PCA: [(batch*time), N*D]
        gt_flat = ground_truth.cpu().reshape(-1, embedding_dim)
        sampled_flat = sampled.cpu().reshape(-1, embedding_dim)

        # Apply PCA on the combined flattened data
        try:
            pca = PCA(n_components=n_components)
            # Fit PCA on ground truth, transform both
            gt_transformed_flat = pca.fit_transform(gt_flat)
            sampled_transformed_flat = pca.transform(sampled_flat)

            # Reshape back to [batch, time, n_components]
            gt_pca = gt_transformed_flat.reshape(n_samples, n_timesteps, n_components)
            sampled_pca = sampled_transformed_flat.reshape(n_samples, n_timesteps, n_components)
        except Exception as e:
            logger.error(f"PCA failed for {n_components}D plot: {e}")
            return

        # Create plot
        fig = plt.figure(figsize=(14, 12) if n_components == 3 else (12, 10))
        ax = fig.add_subplot(111, projection='3d' if n_components == 3 else None)
        colors = plt.cm.viridis(np.linspace(0, 1, n_samples)) # Use viridis for better contrast

        for i in range(n_samples):
            # PCA coordinates for sample i
            gt_coords = gt_pca[i]      # Shape [2, n_components]
            sampled_coords = sampled_pca[i] # Shape [2, n_components]

            if n_components == 2:
                # Plot GT trajectory (dashed)
                ax.plot(gt_coords[:, 0], gt_coords[:, 1], '--', color=colors[i], alpha=0.7, linewidth=1.5, label='Ground Truth' if i == 0 else "")
                # Plot Sampled trajectory (solid)
                ax.plot(sampled_coords[:, 0], sampled_coords[:, 1], '-', color=colors[i], alpha=0.9, linewidth=2, label='Sampled' if i == 0 else "")

                # Mark points: Start (o), GT End (x), Sampled End (+)
                ax.scatter(gt_coords[0, 0], gt_coords[0, 1], color='blue', s=60, marker='o', label='Start' if i == 0 else "")
                ax.scatter(gt_coords[1, 0], gt_coords[1, 1], color='green', s=70, marker='x', label='GT End' if i == 0 else "")
                ax.scatter(sampled_coords[1, 0], sampled_coords[1, 1], color='red', s=70, marker='+', label='Sampled End' if i == 0 else "")
            else: # n_components == 3
                 # Plot GT trajectory (dashed)
                ax.plot(gt_coords[:, 0], gt_coords[:, 1], gt_coords[:, 2], '--', color=colors[i], alpha=0.7, linewidth=1.5, label='Ground Truth' if i == 0 else "")
                # Plot Sampled trajectory (solid)
                ax.plot(sampled_coords[:, 0], sampled_coords[:, 1], sampled_coords[:, 2], '-', color=colors[i], alpha=0.9, linewidth=2, label='Sampled' if i == 0 else "")

                # Mark points: Start (o), GT End (x), Sampled End (+)
                ax.scatter(gt_coords[0, 0], gt_coords[0, 1], gt_coords[0, 2], color='blue', s=60, marker='o', label='Start' if i == 0 else "")
                ax.scatter(gt_coords[1, 0], gt_coords[1, 1], gt_coords[1, 2], color='green', s=70, marker='x', label='GT End' if i == 0 else "")
                ax.scatter(sampled_coords[1, 0], sampled_coords[1, 1], sampled_coords[1, 2], color='red', s=70, marker='+', label='Sampled End' if i == 0 else "")


        # Titles and labels
        variance_ratios = pca.explained_variance_ratio_
        variance_str = ", ".join([f"PC{j+1}={var:.1%}" for j, var in enumerate(variance_ratios)])
        ax.set_title(f'Flow Trajectories - {n_components}D PCA Projection (Step {step})\nExplained Variance: {variance_str}')
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        if n_components == 3:
            ax.set_zlabel('Third Principal Component')
            
        ax.grid(True, alpha=0.3)
        ax.legend() # Display labels set for the first sample

        # Save plot
        plot_filename = (PLOT_2D_FILENAME_TEMPLATE if n_components == 2 else PLOT_3D_FILENAME_TEMPLATE).format(step)
        plot_path = os.path.join(output_dir, plot_filename)
        try:
            plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
            logger.info(f"Saved {n_components}D PCA plot to {plot_path}")
        except IOError as e:
            logger.error(f"Failed to save {n_components}D PCA plot to {plot_path}: {e}")
        finally:
            plt.close(fig) # Close the figure to free memory


    def save_model(self, path: str):
        """Saves the model and optimizer state dictionaries."""
        logger.info(f"Saving model checkpoint to {path}")
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                # Add other info if needed, e.g., epoch, best_val_loss
            }, path)
        except IOError as e:
            logger.error(f"Failed to save model to {path}: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during model saving: {e}")


    def load_model(self, path: str):
        """Loads the model and optimizer state dictionaries."""
        logger.info(f"Loading model checkpoint from {path}")
        if not os.path.exists(path):
             logger.error(f"Checkpoint file not found at {path}. Cannot load model.")
             raise FileNotFoundError(f"Checkpoint file not found: {path}")
        try:
            # Load checkpoint onto the model's current device
            checkpoint = torch.load(path, map_location=self.device)

            if 'model_state_dict' not in checkpoint:
                logger.error(f"Checkpoint {path} missing 'model_state_dict'.")
                raise KeyError("Checkpoint missing 'model_state_dict'.")
            if 'optimizer_state_dict' not in checkpoint:
                logger.warning(f"Checkpoint {path} missing 'optimizer_state_dict'. Optimizer state will not be restored.")
                # Adjust loading if optimizer state is optional for continued training
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                 self.model.load_state_dict(checkpoint['model_state_dict'])
                 self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            logger.info(f"Model and optimizer states loaded successfully from {path}.")
            # Optionally load and return other saved info like epoch number
            # epoch = checkpoint.get('epoch', 0)
            # return epoch

        except FileNotFoundError:
             # Already handled above, but keep for clarity
             raise
        except (KeyError, RuntimeError, IOError) as e:
            logger.error(f"Failed to load model checkpoint from {path}: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during model loading: {e}")
            raise


def prepare_data(
    trajectories: np.ndarray,
    train_ratio: float = DEFAULT_TRAIN_RATIO
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits trajectory data into training and validation sets.

    Performs a simple split based on the train_ratio without shuffling
    to potentially preserve temporal or related characteristics if desired.
    If shuffling is needed before split, it should happen beforehand.

    Args:
        trajectories: NumPy array of trajectories [num_samples, ...].
        train_ratio: Proportion of data to use for training (0.0 to 1.0).

    Returns:
        A tuple containing (train_data, validation_data).
    """
    n_samples = trajectories.shape[0]
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1 (exclusive).")

    n_train = int(n_samples * train_ratio)
    n_val = n_samples - n_train

    if n_train == 0 or n_val == 0:
        logger.warning(f"Data split resulted in zero samples for train ({n_train}) or validation ({n_val}). Adjust train_ratio or data size.")
        # Depending on strictness, could raise error or return empty arrays
        # raise ValueError("Data split resulted in an empty train or validation set.")
    
    train_data = trajectories[:n_train]
    val_data = trajectories[n_train:]
    
    logger.info(f"Data split: {n_train} training samples, {n_val} validation samples.")
    return train_data, val_data


def train_flow_model(
    trajectories: np.ndarray,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    patience: int = DEFAULT_PATIENCE,
    validation_interval: int = VALIDATION_INTERVAL,
    visualization_interval: int = VISUALIZATION_INTERVAL,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
) -> Tuple[FlowMatching, Dict[str, List[float]]]:
    """
    Main training loop for the Conditional Flow Matching model.

    Features include:
    - Data splitting.
    - Model initialization.
    - ExponentialLR learning rate scheduling.
    - Checkpoint loading/saving.
    - Training and validation loops.
    - Early stopping based on validation loss.
    - Periodic visualization.
    - Loss curve plotting.

    Args:
        trajectories: NumPy array of trajectory data [num_samples, 2, N, D].
        output_dir: Directory to save models, logs, and visualizations.
        num_epochs: Maximum number of training epochs.
        batch_size: Number of samples per training batch.
        lr: Maximum learning rate for the scheduler.
        weight_decay: Weight decay for the AdamW optimizer.
        patience: Number of validation checks to wait for improvement before early stopping.
        validation_interval: Frequency (in epochs) to perform validation.
        visualization_interval: Frequency (in epochs) to generate visualizations.
        train_ratio: Fraction of data used for training.

    Returns:
        A tuple containing:
            - The trained FlowMatching model (loaded with best weights).
            - A dictionary containing lists of 'train' and 'val' losses per epoch/validation step.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Starting training process. Output directory: {output_dir}")

    # 1. Prepare Data
    logger.info("Preparing data...")
    try:
        train_data, val_data = prepare_data(trajectories, train_ratio=train_ratio)
    except ValueError as e:
        logger.error(f"Data preparation failed: {e}")
        raise # Re-raise to stop execution

    if len(train_data) == 0:
         logger.error("Training dataset is empty after split. Cannot train.")
         raise ValueError("Training dataset is empty.")


    # 2. Initialize Model and Optimizer
    logger.info("Initializing FlowMatching model...")
    flow_model = FlowMatching(lr=lr, weight_decay=weight_decay)

    # 3. Initialize Scheduler
    # Calculate steps considering potential partial last batch
    steps_per_epoch = (len(train_data) + batch_size - 1) // batch_size
    logger.info(f"Scheduler: ExponentialLR (gamma=0.99), Initial LR={lr:.2e}, Steps/epoch: {steps_per_epoch}")
    scheduler = ExponentialLR(flow_model.optimizer, gamma=0.99)

    # 4. Load Existing Checkpoint (if available)
    final_model_path = os.path.join(output_dir, FINAL_MODEL_FILENAME)
    best_model_path = os.path.join(output_dir, BEST_MODEL_FILENAME)
    start_epoch = 0
    if os.path.exists(final_model_path): # Check for final model to resume
        try:
            logger.warning(f"Found existing model at {final_model_path}. Attempting to resume training.")
            # Consider loading epoch number from checkpoint if saved
            flow_model.load_model(final_model_path)
            logger.info("Resumed training from the last saved state.")
            # If epoch was saved: start_epoch = loaded_epoch + 1
            # If LR state needs resuming, handle scheduler state loading here too.
        except Exception as e:
            logger.error(f"Could not load model from {final_model_path}: {e}. Starting fresh.", exc_info=True)
            # Re-initialize model if loading fails? Or just proceed with the initial one.
            flow_model = FlowMatching(lr=lr, weight_decay=weight_decay) # Re-init if needed
    else:
        logger.info("No existing model found. Starting training from scratch.")


    # 5. Training Loop Setup
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    logger.info(f"Starting training loop for {num_epochs} epochs (batch size: {batch_size}).")
    logger.info(f"Validation every {validation_interval} epochs. Early stopping patience: {patience}.")
    if len(val_data) > 0:
        logger.info(f"Visualization every {visualization_interval} epochs on validation subset.")
    else:
        logger.warning("No validation data. Validation, early stopping, and visualization will be skipped.")


    # 6. Training Loop
    for epoch in range(start_epoch, num_epochs):
        epoch_num = epoch + 1
        flow_model.model.train() # Set model to training mode
        epoch_train_losses = []

        # Use tqdm for progress bar over batches
        batch_iterator = tqdm(range(steps_per_epoch),
                              desc=f"Epoch {epoch_num}/{num_epochs} Training",
                              leave=False) # Keep progress bar until epoch ends

        for i in batch_iterator:
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(train_data))
            # This check might be redundant if steps_per_epoch is correct
            if start_idx >= end_idx: continue

            # Get batch and move to device
            batch_np = train_data[start_idx:end_idx]
            batch_tensor = torch.tensor(batch_np, dtype=torch.float32, device=flow_model.device)

            actual_batch_size = batch_tensor.shape[0]
            if actual_batch_size == 0: continue

            x0, x1 = batch_tensor[:, 0], batch_tensor[:, -1] # [B, N, D]

            # Stratified time sampling for the batch
            time_offset = torch.rand(1, device=flow_model.device) # Single offset for the batch
            time_steps = torch.arange(actual_batch_size, device=flow_model.device) / actual_batch_size
            t = (time_offset + time_steps) % (1.0 - TIME_SAMPLING_EPS)
            t = t.unsqueeze(1) # [B, 1]

            # Perform training step
            loss = flow_model.train_step(x0, x1, t)
            epoch_train_losses.append(loss)

            # Update tqdm description with current loss
            batch_iterator.set_postfix(loss=f"{loss:.4f}", lr=f"{flow_model.optimizer.param_groups[0]['lr']:.2e}")


        # --- End of Epoch ---
        avg_epoch_loss = np.mean(epoch_train_losses) if epoch_train_losses else 0
        train_losses.append(avg_epoch_loss)

        # Update learning rate scheduler at the end of the epoch
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0] # Get LR after step
        log_msg = f"Epoch {epoch_num}/{num_epochs} | Avg Train Loss: {avg_epoch_loss:.6f} | LR: {current_lr:.6e}"

        # 7. Validation (if applicable and interval reached)
        perform_validation = (len(val_data) > 0) and (epoch_num % validation_interval == 0 or epoch_num == num_epochs)
        if perform_validation:
            logger.info(f"--- Starting Validation for Epoch {epoch_num} ---")
            avg_val_loss = validate_model(flow_model, val_data, batch_size)
            val_losses.append(avg_val_loss)
            log_msg += f" | Avg Val Loss: {avg_val_loss:.6f}"

            if avg_val_loss < best_val_loss:
                logger.info(f"Validation loss improved ({best_val_loss:.6f} -> {avg_val_loss:.6f}). Saving best model.")
                best_val_loss = avg_val_loss
                flow_model.save_model(best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                log_msg += f" | Patience: {patience_counter}/{patience}"
                if patience_counter >= patience:
                    logger.warning(f"Early stopping triggered at epoch {epoch_num} due to no validation improvement.")
                    break # Exit training loop
        else:
            # Log training loss even if not validating this epoch
            pass # Log message already contains training info

        logger.info(log_msg) # Log combined epoch summary


        # 8. Visualization (if applicable and interval reached)
        perform_visualization = (len(val_data) > 0) and (epoch_num % visualization_interval == 0 or epoch_num == num_epochs or epoch == start_epoch)
        if perform_visualization:
            logger.info(f"--- Generating Visualization for Epoch {epoch_num} ---")
            # Visualize on a subset of validation data
            viz_subset_size = min(MAX_VIZ_SAMPLES, len(val_data))
            if viz_subset_size > 0:
                flow_model.visualize(
                    val_data[:viz_subset_size],
                    step=epoch_num,
                    output_dir=output_dir,
                    plots=["2d", "3d"],
                    # Use a consistent sampling method for viz, e.g., Euler
                    sampling_method="euler",
                    sampling_steps=1 # Or more steps if desired for viz
                )
            else:
                logger.info("Skipping visualization (subset size is zero).")


    # 9. Post-Training Steps
    logger.info("Training loop finished.")
    logger.info(f"Saving final model state to {final_model_path}")
    flow_model.save_model(final_model_path)

    # Plot loss curves
    _plot_loss_curves(train_losses, val_losses, validation_interval, epoch + 1, output_dir)


    # Load the best model based on validation performance before returning
    if os.path.exists(best_model_path):
        logger.info(f"Loading best model from {best_model_path} based on validation loss.")
        try:
            flow_model.load_model(best_model_path)
        except Exception as e:
            logger.error(f"Failed to load best model: {e}. Returning the final model instead.", exc_info=True)
            # Fallback: ensure the final model is loaded if best fails
            if os.path.exists(final_model_path):
                try:
                    flow_model.load_model(final_model_path)
                except Exception as e_final:
                     logger.critical(f"Failed to load even the final model: {e_final}. Returning potentially untrained model.", exc_info=True)

    else:
        logger.warning("Best model checkpoint not found (possibly no validation data or no improvement). Returning the final model.")
        # Ensure the final model state is loaded if best doesn't exist
        if os.path.exists(final_model_path):
             try:
                flow_model.load_model(final_model_path)
             except Exception as e_final:
                 logger.critical(f"Failed to load the final model: {e_final}. Returning potentially untrained model.", exc_info=True)


    loss_history = {'train': train_losses, 'val': val_losses}
    return flow_model, loss_history


def validate_model(
    flow_model: FlowMatching,
    val_data: np.ndarray,
    batch_size: int = DEFAULT_BATCH_SIZE
) -> float:
    """
    Evaluates the model's performance on the validation dataset.

    Calculates the average Flow Matching MSE loss across the validation set
    using stratified time sampling, similar to training.

    Args:
        flow_model: The FlowMatching model instance.
        val_data: NumPy array of validation trajectories [num_samples, 2, N, D].
        batch_size: Batch size for validation evaluation.

    Returns:
        The average validation loss. Returns float('inf') if val_data is empty.
    """
    if len(val_data) == 0:
        logger.warning("Validation data is empty. Cannot calculate validation loss.")
        return float('inf') # Or 0, depending on desired behavior

    flow_model.model.eval() # Set model to evaluation mode
    val_losses = []
    num_val_samples = len(val_data)
    num_batches = (num_val_samples + batch_size - 1) // batch_size

    logger.info(f"Processing {num_val_samples} validation samples in {num_batches} batches.")

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Validation Progress", leave=False):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_val_samples)
            if start_idx >= end_idx: continue

            batch_np = val_data[start_idx:end_idx]
            batch_tensor = torch.tensor(batch_np, dtype=torch.float32, device=flow_model.device)

            actual_batch_size = batch_tensor.shape[0]
            if actual_batch_size == 0: continue

            x0, x1 = batch_tensor[:, 0], batch_tensor[:, -1] # [B, N, D]

            # Stratified time sampling for validation consistency
            # Use fixed seed or random offset per batch? Using random offset here.
            time_offset = torch.rand(1, device=flow_model.device)
            time_steps = torch.arange(actual_batch_size, device=flow_model.device) / actual_batch_size
            t = (time_offset + time_steps) % (1.0 - TIME_SAMPLING_EPS)
            t = t.unsqueeze(1) # [B, 1]

            # Calculate loss (same as training logic but without backprop)
            t_broadcast = t.view(actual_batch_size, 1, 1)
            ut_target = x1 - x0
            xt_sample = x0 + t_broadcast * ut_target
            ut_predicted = flow_model.model(xt_sample, t)
            loss = F.mse_loss(ut_predicted, ut_target)
            val_losses.append(loss.item())

    avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
    logger.info(f"Validation finished. Average loss: {avg_val_loss:.6f}")
    return avg_val_loss


def load_trajectory_data(data_dir: str = DEFAULT_DATA_DIR, file_pattern: str = DATA_FILE_PATTERN) -> np.ndarray:
    """
    Loads and combines trajectory data (x0, x1) from multiple NPZ files.

    Performs validation checks on file contents and shapes, pre-allocates memory,
    loads data efficiently, and shuffles the combined trajectories.

    Args:
        data_dir: Directory containing the NPZ files.
        file_pattern: Glob pattern to match trajectory files (e.g., 'embeddings_*.npz').

    Returns:
        A NumPy array of combined and shuffled trajectories with shape
        [total_valid_samples, 2, N, D].

    Raises:
        ValueError: If no valid NPZ files are found or processed.
        MemoryError: If allocation of the final array fails.
        FileNotFoundError: If the data_dir does not exist.
    """
    if not os.path.isdir(data_dir):
         logger.error(f"Data directory not found: {data_dir}")
         raise FileNotFoundError(f"Data directory not found: {data_dir}")

    full_pattern = os.path.join(data_dir, file_pattern)
    file_paths = sorted(glob.glob(full_pattern))
    logger.info(f"Searching for trajectory files matching '{full_pattern}'. Found {len(file_paths)} files.")

    if not file_paths:
        logger.error(f"No NPZ files found matching the pattern in {data_dir}.")
        raise ValueError(f"No NPZ files found matching '{file_pattern}' in {data_dir}")

    # --- First Pass: Metadata Scan ---
    total_samples = 0
    expected_shape_inner = None # (N, D)
    dtype = None
    valid_files = []
    logger.info("Scanning files to determine total size, shape, and dtype...")

    for i, file_path in enumerate(file_paths):
        try:
            with np.load(file_path, allow_pickle=False) as data: # Disable pickle for security
                if 'x0' not in data or 'x1' not in data:
                    logger.warning(f"Skipping {os.path.basename(file_path)}: missing 'x0' or 'x1' key.")
                    continue

                x0_data = data['x0']
                x1_data = data['x1']
                x0_shape = x0_data.shape
                x1_shape = x1_data.shape

                # Basic checks
                if x0_shape[0] != x1_shape[0]:
                    logger.warning(f"Skipping {os.path.basename(file_path)}: Sample count mismatch (x0: {x0_shape[0]}, x1: {x1_shape[0]}).")
                    continue
                current_samples = x0_shape[0]
                if current_samples == 0:
                    logger.warning(f"Skipping {os.path.basename(file_path)}: Contains 0 samples.")
                    continue

                # Shape validation (assuming [N, D] or [Batch, N, D] format inside npz)
                # We expect the data loader to yield [Batch, 2, N, D] later.
                # The saved format seems to be [Batch_in_file, N, D] for x0 and x1.
                if len(x0_shape) != 3 or len(x1_shape) != 3:
                     logger.warning(f"Skipping {os.path.basename(file_path)}: Unexpected data dimension. Expected 3D array (Batch, N, D), got x0: {len(x0_shape)}D, x1: {len(x1_shape)}D.")
                     continue

                # Check inner shape (N, D) consistency
                current_inner_shape = x0_shape[1:]
                if x1_shape[1:] != current_inner_shape:
                     logger.warning(f"Skipping {os.path.basename(file_path)}: Inner shape mismatch (x0: {x0_shape[1:]}, x1: {x1_shape[1:]}).")
                     continue

                current_dtype = x0_data.dtype

                # Initialize expected shape and dtype from the first valid file
                if expected_shape_inner is None:
                    expected_shape_inner = current_inner_shape # e.g., (49, 128)
                    dtype = current_dtype
                    logger.info(f"Determined expected inner shape: {expected_shape_inner}, dtype: {dtype} from {os.path.basename(file_path)}")
                # Check subsequent files against the first valid one
                elif current_inner_shape != expected_shape_inner:
                    logger.warning(f"Skipping {os.path.basename(file_path)}: Inner shape {current_inner_shape} mismatches expected {expected_shape_inner}.")
                    continue
                elif current_dtype != dtype:
                    logger.warning(f"Skipping {os.path.basename(file_path)}: dtype {current_dtype} mismatches expected {dtype}.")
                    continue

                # If all checks pass
                total_samples += current_samples
                valid_files.append(file_path)

        except Exception as e:
            logger.error(f"Error reading metadata from {os.path.basename(file_path)}: {e}. Skipping.", exc_info=True)
            continue

        # Log progress occasionally
        if (i + 1) % 50 == 0:
            logger.info(f"Scanned {i+1}/{len(file_paths)} files. Current valid samples: {total_samples}")

    if total_samples == 0 or expected_shape_inner is None:
        logger.error("Scan complete, but no valid trajectory data could be processed.")
        raise ValueError("No valid trajectory data found or processed.")

    logger.info(f"Scan complete. Total valid samples: {total_samples}. Using {len(valid_files)} files.")

    # --- Pre-allocation ---
    # Final shape will be [total_samples, 2 (time), N, D]
    final_shape = (total_samples, 2) + expected_shape_inner
    logger.info(f"Attempting to allocate NumPy array with shape: {final_shape}, dtype: {dtype}")
    try:
        combined_trajectories = np.zeros(final_shape, dtype=dtype)
        logger.info("Memory allocation successful.")
    except MemoryError as e:
        logger.error(f"MemoryError: Failed to allocate array of shape {final_shape}. Available memory might be insufficient.")
        raise e # Re-raise MemoryError
    except Exception as e:
        logger.error(f"Unexpected error during array allocation: {e}")
        raise # Re-raise other unexpected errors

    # --- Second Pass: Data Loading ---
    current_pos = 0
    logger.info(f"Loading data from {len(valid_files)} valid files...")
    for file_path in tqdm(valid_files, desc="Loading Trajectory Files"):
        try:
            with np.load(file_path) as data:
                x0 = data['x0'] # [Batch_in_file, N, D]
                x1 = data['x1'] # [Batch_in_file, N, D]
                num_samples_in_file = x0.shape[0]

                # Define slice to fill
                start_idx = current_pos
                end_idx = current_pos + num_samples_in_file

                # Safety check (should not happen if first pass was correct)
                if end_idx > total_samples:
                    logger.warning(f"Data from {os.path.basename(file_path)} exceeds allocated space ({end_idx} > {total_samples}). Truncating load.")
                    num_samples_to_load = total_samples - start_idx
                    if num_samples_to_load <= 0: continue # No space left
                    end_idx = total_samples
                    x0 = x0[:num_samples_to_load]
                    x1 = x1[:num_samples_to_load]
                else:
                     num_samples_to_load = num_samples_in_file


                # Place data into the pre-allocated array
                combined_trajectories[start_idx:end_idx, 0] = x0
                combined_trajectories[start_idx:end_idx, 1] = x1

                current_pos = end_idx

        except Exception as e:
            logger.error(f"Error loading data from {os.path.basename(file_path)} during second pass: {e}. Skipping file, data might be incomplete.", exc_info=True)
            # If an error occurs, current_pos will not advance correctly relative to total_samples

    # Final check on loaded size
    if current_pos != total_samples:
        logger.warning(f"Data loading finished, but loaded samples ({current_pos}) do not match expected total ({total_samples}). This might indicate errors during loading. Trimming array.")
        combined_trajectories = combined_trajectories[:current_pos]

    if len(combined_trajectories) == 0:
        logger.error("Combined trajectory array is empty after loading.")
        raise ValueError("Failed to load any data into the combined array.")

    logger.info(f"Successfully loaded {len(combined_trajectories)} samples.")

    # --- Shuffle Data ---
    logger.info(f"Shuffling {len(combined_trajectories)} trajectories...")
    np.random.seed(42) # Ensure reproducibility
    shuffle_indices = np.random.permutation(len(combined_trajectories))
    combined_trajectories = combined_trajectories[shuffle_indices]
    logger.info("Data loading and shuffling complete.")

    return combined_trajectories


def _plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    validation_interval: int,
    total_epochs: int,
    output_dir: str
):
    """Helper function to plot training and validation loss curves."""
    logger.info("Plotting loss curves...")
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Progress')
    plt.grid(True, alpha=0.3)
    plt.legend()
    train_loss_path = os.path.join(output_dir, TRAIN_LOSS_PLOT_FILENAME)
    try:
        plt.savefig(train_loss_path)
        logger.info(f"Training loss curve saved to {train_loss_path}")
    except IOError as e:
        logger.error(f"Failed to save training loss plot: {e}")
    finally:
        plt.close()

    if val_losses:
        # Validation runs every `validation_interval` epochs.
        # Create corresponding epoch numbers for the validation losses.
        # Example: If interval=10, epochs are 10, 20, 30...
        val_epochs = list(range(validation_interval, total_epochs + 1, validation_interval))

        # Adjust for early stopping: ensure lengths match
        num_val_points = len(val_losses)
        val_epochs_adjusted = val_epochs[:num_val_points]

        plt.figure(figsize=(12, 6))
        plt.plot(val_epochs_adjusted, val_losses, 'r-', label='Validation Loss', linewidth=2)
        # Optional: Plot training loss on the same graph for comparison
        # plt.plot(epochs, train_losses, 'b--', label='Training Loss', alpha=0.5)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss Progress')
        plt.grid(True, alpha=0.3)
        plt.legend()
        val_loss_path = os.path.join(output_dir, VAL_LOSS_PLOT_FILENAME)
        try:
            plt.savefig(val_loss_path)
            logger.info(f"Validation loss curve saved to {val_loss_path}")
        except IOError as e:
            logger.error(f"Failed to save validation loss plot: {e}")
        finally:
            plt.close()
    else:
        logger.info("Skipping validation loss plot (no validation data).")


def setup_logging(log_file_path: str):
    """Configures logging to file and console."""
    log_dir = os.path.dirname(log_file_path)
    if log_dir: # Ensure directory exists if specified
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file_path, mode='w'), # Overwrite log file
            logging.StreamHandler(sys.stdout) # Log to console
        ]
    )
    # Get the root logger used by basicConfig
    root_logger = logging.getLogger()
    # Optionally set level for specific libraries if too verbose
    # logging.getLogger('matplotlib').setLevel(logging.WARNING)
    return root_logger


if __name__ == "__main__":
    """
    Main script execution block for training the Flow Matching model.

    1. Sets up configuration (paths, hyperparameters).
    2. Configures logging.
    3. Loads trajectory data.
    4. Initiates the training process.
    5. Handles potential errors during execution.
    6. Performs final visualization using the best trained model.
    """
    # --- Configuration ---
    output_dir = DEFAULT_OUTPUT_DIR
    data_dir = DEFAULT_DATA_DIR
    num_epochs = DEFAULT_NUM_EPOCHS
    batch_size = DEFAULT_BATCH_SIZE
    learning_rate = DEFAULT_LR
    weight_decay = DEFAULT_WEIGHT_DECAY
    patience = DEFAULT_PATIENCE
    validation_interval = VALIDATION_INTERVAL
    visualization_interval = VISUALIZATION_INTERVAL
    train_ratio = DEFAULT_TRAIN_RATIO

    # --- Logging Setup ---
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, TRAINING_LOG_FILENAME)
    logger = setup_logging(log_file) # Use the setup function

    logger.info("--- Starting Training Script ---")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Log File: {log_file}")
    logger.info(f"Data Directory: {data_dir}")
    current_device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using Device: {current_device}")
    logger.info(f"Training Parameters: Epochs={num_epochs}, BatchSize={batch_size}, LR={learning_rate:.1e}, WeightDecay={weight_decay:.1e}")
    logger.info(f"Validation: Interval={validation_interval}, Patience={patience}")
    logger.info(f"Visualization: Interval={visualization_interval}")

    # --- Load Data ---
    logger.info("Loading trajectory data...")
    try:
        trajectories = load_trajectory_data(data_dir=data_dir)
        logger.info(f"Loaded {len(trajectories)} trajectories with shape {trajectories.shape}")
    except (FileNotFoundError, ValueError, MemoryError) as e:
        logger.error(f"Fatal Error: Failed to load data. {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.exception("An unexpected error occurred during data loading.")
        sys.exit(1)

    # --- Train Model ---
    logger.info("Starting model training...")
    try:
        flow_model, loss_history = train_flow_model(
            trajectories=trajectories,
            output_dir=output_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
            validation_interval=validation_interval,
            visualization_interval=visualization_interval,
            train_ratio=train_ratio,
        )
        logger.info("Model training finished successfully.")
    except (ValueError, RuntimeError, MemoryError) as e: # Catch specific training errors
        logger.error(f"Fatal Error: Model training failed. {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.exception("An unexpected error occurred during model training.")
        sys.exit(1)

    # --- Final Visualization (using the best loaded model) ---
    logger.info("Generating final visualizations using the best trained model...")
    try:
        # Need validation data again for visualization context
        # It's generally better to visualize based on a fixed validation set
        _, val_data = prepare_data(trajectories, train_ratio=train_ratio)
        if len(val_data) > 0:
            viz_subset_size = min(20, len(val_data)) # Visualize up to 20 samples
            flow_model.visualize(
                val_data[:viz_subset_size],
                step='final', # Special step identifier
                output_dir=output_dir,
                plots=["2d", "3d"],
                sampling_method="euler", # Use a consistent method
                sampling_steps=1
            )
            logger.info("Final visualizations saved.")
        else:
            logger.info("Skipping final visualization (no validation data).")
    except Exception as e:
        logger.exception("An unexpected error occurred during final visualization.")
        # Don't necessarily exit here, training might still be successful

    logger.info("--- Training script completed ---")