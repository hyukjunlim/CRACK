import logging
import os
import sys
import time
import glob
from typing import Tuple, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Assuming these are available in the environment
try:
    from nets.equiformer_v2.MPFlow import EquivariantMPFlow
    from nets.equiformer_v2.so3 import SO3_Grid
    from nets.equiformer_v2.module_list import ModuleListInfo
except ImportError as e:
    logging.error(f"Failed to import required network components: {e}")
    sys.exit(1)

# Optional 3D plotting
try:
    from mpl_toolkits.mplot3d import Axes3D
    MPL_3D_AVAILABLE = True
except ImportError:
    Axes3D = None # Define Axes3D as None if import fails
    MPL_3D_AVAILABLE = False
    logging.warning("mpl_toolkits.mplot3d not found. 3D PCA plot will be disabled.")


# --- Constants (Shared structure with train script, consider common file) ---
# Data Shape (Assuming consistency with training data)
TIME_DIM = 2
NODE_DIM = 49
FEATURE_DIM = 128
EMBEDDING_SHAPE = (NODE_DIM, FEATURE_DIM)
TRAJECTORY_SHAPE = (TIME_DIM,) + EMBEDDING_SHAPE # (2, 49, 128)

# Model Hyperparameters (Must match the trained model)
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

# Prediction & Visualization
DEFAULT_INFERENCE_BATCH_SIZE = 1024 # Adjusted default
DEFAULT_EULER_STEPS = 1
SUPPORTED_SAMPLING_METHODS = ["heun", "rk4", "euler"]
PCA_N_COMPONENTS_2D = 2
PCA_N_COMPONENTS_3D = 3
MAX_VIZ_SAMPLES = 8
PLOT_DPI = 300

# File Paths & Names
DEFAULT_MODEL_DIR = 'flow_output/exp2' # Match default train output
BEST_MODEL_FILENAME = 'best_flow_model.pt'
DEFAULT_DATA_DIR = 'mpflow_data_small' # Use full dataset for prediction? Or specify test set dir
DATA_FILE_PATTERN = 'embeddings_*.npz'
DEFAULT_RESULTS_DIR = DEFAULT_MODEL_DIR # Save results alongside model
RESULTS_FILENAME = "inference_results.txt"
VISUALIZATION_SUBDIR = "prediction_visualizations"
PLOT_2D_FILENAME_TEMPLATE = "prediction_pca_2d_{method}_{n_steps}steps.png"
PLOT_3D_FILENAME_TEMPLATE = "prediction_pca_3d_{method}_{n_steps}steps.png"

# Root logger for the script
logger = logging.getLogger(__name__) # Use named logger

class FlowPredictor:
    """
    Manages loading a pre-trained Flow Matching model and performing inference.

    Handles model loading, trajectory sampling (predicting x1 from x0),
    calculating performance metrics (MSE, timing), and visualizing predictions
    against ground truth.
    """
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = torch.device(device)
        logger.info(f"Initializing FlowPredictor on device: {self.device}")

        # --- Model Configuration (Must match trained model) ---
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
        # Optimizer is not needed for prediction

    def _build_so3_grid(self) -> ModuleListInfo:
        """Builds the SO3 grid structure (identical to training)."""
        max_l = max(self.lmax_list)
        so3_grid = ModuleListInfo(f'({max_l}, {max_l})')
        for l in range(max_l + 1):
            so3_m_grid = nn.ModuleList()
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
        """Builds the EquivariantMPFlow model (identical to training)."""
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
        # Log parameter count during initialization, before loading weights
        logger.info(f"EquivariantMPFlow model structure created with {sum(p.numel() for p in model.parameters()):,} parameters.")
        return model

    def load_model_weights(self, path: str):
        """Loads only the model state dictionary from a checkpoint."""
        logger.info(f"Loading model weights from checkpoint: {path}")
        if not os.path.exists(path):
             logger.error(f"Checkpoint file not found at {path}. Cannot load weights.")
             raise FileNotFoundError(f"Checkpoint file not found: {path}")
        try:
            # Load checkpoint onto the correct device directly
            checkpoint = torch.load(path, map_location=self.device)

            if 'model_state_dict' not in checkpoint:
                logger.error(f"Checkpoint {path} missing 'model_state_dict'.")
                raise KeyError("Checkpoint missing 'model_state_dict'.")

            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Model weights loaded successfully.")
            self.model.eval() # Ensure model is in eval mode after loading weights
            logger.info("Model set to evaluation mode.")

        except FileNotFoundError:
             raise # Re-raise specific error
        except (KeyError, RuntimeError, IOError) as e:
            logger.error(f"Failed to load model weights from {path}: {e}")
            raise # Re-raise other loading errors
        except Exception as e:
            logger.error(f"An unexpected error occurred during weight loading: {e}")
            raise # Re-raise unexpected errors

    def predict_final_state(
        self,
        x0: torch.Tensor,
        method: str = "heun",
        n_steps: int = 1
    ) -> torch.Tensor:
        """
        Predicts the final state x1_pred given the initial state x0.

        Uses the loaded model and a specified ODE solver method.

        Args:
            x0: Starting point tensor [batch, N, D].
            method: Sampling method ("heun", "rk4", "euler").
            n_steps: Number of integration steps (only used for "euler").

        Returns:
            The predicted final state tensor x1_pred [batch, N, D].

        Raises:
            ValueError: If an unsupported sampling method is provided.
        """
        if method not in SUPPORTED_SAMPLING_METHODS:
            logger.error(f"Unsupported sampling method: {method}")
            raise ValueError(f"Unsupported sampling method: {method}. Supported: {SUPPORTED_SAMPLING_METHODS}")

        self.model.eval() # Ensure model is in evaluation mode
        batch_size = x0.shape[0]
        device = x0.device

        x_current = x0.clone() # Work on a copy

        with torch.no_grad():
            if method == "heun":
                # Single-step Heun's method (Predictor-Corrector) from t=0 to t=1
                t0 = torch.zeros((batch_size, 1), device=device)
                t1 = torch.ones((batch_size, 1), device=device)
                v0 = self.model(x_current, t0)
                x_pred_euler = x_current + v0 # h=1
                v1_pred = self.model(x_pred_euler, t1)
                x_final = x_current + 0.5 * (v0 + v1_pred) # h=1

            elif method == "rk4":
                # Single-step Runge-Kutta 4th order from t=0 to t=1
                t0 = torch.zeros((batch_size, 1), device=device)
                t_half = torch.full((batch_size, 1), 0.5, device=device)
                t1 = torch.ones((batch_size, 1), device=device)
                k1 = self.model(x_current, t0)
                k2 = self.model(x_current + 0.5 * k1, t_half) # h=1 assumed
                k3 = self.model(x_current + 0.5 * k2, t_half) # h=1 assumed
                k4 = self.model(x_current + k3, t1)       # h=1 assumed
                x_final = x_current + (1.0 / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

            elif method == "euler":
                # Multi-step Forward Euler method
                if n_steps <= 0:
                    logger.warning(f"Euler method called with n_steps={n_steps}. Performing single step.")
                    n_steps = 1
                dt = 1.0 / n_steps
                x_t = x_current
                for i in range(n_steps):
                    t_current = torch.full((batch_size, 1), i * dt, device=device)
                    velocity = self.model(x_t, t_current)
                    x_t = x_t + dt * velocity
                x_final = x_t

        return x_final # Return only the predicted final state [batch, N, D]


    def visualize_predictions(
        self,
        x0: torch.Tensor,
        x1_true: torch.Tensor,
        x1_pred: torch.Tensor,
        method: str,
        n_steps: int,
        output_dir: str,
        plots: List[str] = ["2d", "3d"]
    ):
        """
        Creates visualizations comparing predicted vs ground truth final states.

        Generates 2D/3D PCA plots showing start, true end, and predicted end points.

        Args:
            x0: Start points tensor [batch, N, D].
            x1_true: Ground truth end points tensor [batch, N, D].
            x1_pred: Predicted end points tensor [batch, N, D].
            method: Name of the prediction method used.
            n_steps: Number of steps used (relevant for Euler).
            output_dir: Directory to save plots.
            plots: List of plots to generate ("2d", "3d").
        """
        n_samples = x0.shape[0]
        if n_samples == 0:
            logger.warning("No samples provided for visualization.")
            return

        logger.info(f"Generating prediction visualizations for {n_samples} samples using method '{method}' ({n_steps} steps)...")
        os.makedirs(output_dir, exist_ok=True)

        # Create visualizations
        common_args = (x0, x1_true, x1_pred, method, n_steps, output_dir)
        if "2d" in plots:
            self._create_pca_plot(PCA_N_COMPONENTS_2D, *common_args)

        if "3d" in plots:
            if MPL_3D_AVAILABLE:
                self._create_pca_plot(PCA_N_COMPONENTS_3D, *common_args)
            else:
                logger.warning("Skipping 3D PCA plot because mpl_toolkits.mplot3d is not available.")


    def _create_pca_plot(
        self,
        n_components: int,
        x0: torch.Tensor,
        x1_true: torch.Tensor,
        x1_pred: torch.Tensor,
        method: str,
        n_steps: int,
        output_dir: str
    ):
        """
        Helper to create 2D or 3D PCA plots for prediction visualization.

        Args:
            n_components: Number of PCA components (2 or 3).
            x0, x1_true, x1_pred: Tensors [batch, N, D].
            method: Prediction method name.
            n_steps: Number of steps used.
            output_dir: Save directory.
        """
        n_samples = x0.shape[0]
        embedding_dim = x0.shape[1] * x0.shape[2] # N * D

        # Flatten embedding dimensions: [batch, N*D]
        # Move to CPU and convert to NumPy for PCA
        x0_flat = x0.cpu().numpy().reshape(n_samples, embedding_dim)
        x1_true_flat = x1_true.cpu().numpy().reshape(n_samples, embedding_dim)
        x1_pred_flat = x1_pred.cpu().numpy().reshape(n_samples, embedding_dim)

        # Combine all points for PCA fitting: [3 * batch, N*D]
        all_points_flat = np.vstack((x0_flat, x1_true_flat, x1_pred_flat))

        try:
            # Apply PCA
            pca = PCA(n_components=n_components)
            all_points_transformed = pca.fit_transform(all_points_flat)

            # Separate the transformed points
            x0_pca = all_points_transformed[0*n_samples : 1*n_samples]        # [batch, n_components]
            x1_true_pca = all_points_transformed[1*n_samples : 2*n_samples]   # [batch, n_components]
            x1_pred_pca = all_points_transformed[2*n_samples : 3*n_samples]   # [batch, n_components]

        except Exception as e:
            logger.error(f"PCA failed for {n_components}D plot (method: {method}): {e}", exc_info=True)
            return

        # Create plot
        fig = plt.figure(figsize=(14, 12) if n_components == 3 else (12, 10))
        ax = fig.add_subplot(111, projection='3d' if n_components == 3 else None)
        colors = plt.cm.viridis(np.linspace(0, 1, n_samples))

        for i in range(n_samples):
            # Get PCA coordinates for sample i
            start_coords = x0_pca[i]
            true_end_coords = x1_true_pca[i]
            pred_end_coords = x1_pred_pca[i]

            if n_components == 2:
                # Plot lines: start -> true_end (dashed), start -> pred_end (solid)
                ax.plot([start_coords[0], true_end_coords[0]], [start_coords[1], true_end_coords[1]],
                        '--', color=colors[i], alpha=0.6, linewidth=1.5, label='True Flow' if i==0 else "")
                ax.plot([start_coords[0], pred_end_coords[0]], [start_coords[1], pred_end_coords[1]],
                        '-', color=colors[i], alpha=0.8, linewidth=1.5, label='Predicted Flow' if i==0 else "")

                # Mark points: start (o, blue), true_end (x, green), pred_end (+, red)
                ax.scatter(start_coords[0], start_coords[1], color='blue', s=40, marker='o', label='Start (x0)' if i==0 else "")
                ax.scatter(true_end_coords[0], true_end_coords[1], color='green', s=50, marker='x', label='True End (x1)' if i==0 else "")
                ax.scatter(pred_end_coords[0], pred_end_coords[1], color='red', s=50, marker='+', label='Predicted End' if i==0 else "")
            else: # n_components == 3
                 # Plot lines
                ax.plot([start_coords[0], true_end_coords[0]], [start_coords[1], true_end_coords[1]], [start_coords[2], true_end_coords[2]],
                        '--', color=colors[i], alpha=0.6, linewidth=1.5, label='True Flow' if i==0 else "")
                ax.plot([start_coords[0], pred_end_coords[0]], [start_coords[1], pred_end_coords[1]], [start_coords[2], pred_end_coords[2]],
                        '-', color=colors[i], alpha=0.8, linewidth=1.5, label='Predicted Flow' if i==0 else "")
                 # Mark points
                ax.scatter(start_coords[0], start_coords[1], start_coords[2], color='blue', s=40, marker='o', label='Start (x0)' if i==0 else "")
                ax.scatter(true_end_coords[0], true_end_coords[1], true_end_coords[2], color='green', s=50, marker='x', label='True End (x1)' if i==0 else "")
                ax.scatter(pred_end_coords[0], pred_end_coords[1], pred_end_coords[2], color='red', s=50, marker='+', label='Predicted End' if i==0 else "")


        # Titles and labels
        step_info = f"({n_steps} steps)" if method == "euler" else "(1 step)" # Single step for heun/rk4
        variance_ratios = pca.explained_variance_ratio_
        variance_str = ", ".join([f"PC{j+1}={var:.1%}" for j, var in enumerate(variance_ratios)])

        ax.set_title(f'Prediction vs Truth - {n_components}D PCA ({method.upper()} {step_info})\nExplained Variance: {variance_str}')
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        if n_components == 3:
            ax.set_zlabel('Third Principal Component')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Save plot
        plot_filename_template = PLOT_2D_FILENAME_TEMPLATE if n_components == 2 else PLOT_3D_FILENAME_TEMPLATE
        plot_filename = plot_filename_template.format(method=method, n_steps=n_steps)
        plot_path = os.path.join(output_dir, plot_filename)
        try:
            plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
            logger.info(f"Saved {n_components}D PCA plot to {plot_path}")
        except IOError as e:
            logger.error(f"Failed to save {n_components}D PCA plot to {plot_path}: {e}")
        finally:
            plt.close(fig)


def calculate_prediction_metrics(
    predictor: FlowPredictor,
    data: np.ndarray,
    batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
    method: str = "heun",
    n_steps: int = 1
) -> Tuple[float, float, float]:
    """
    Calculates inference time and Mean Squared Error (MSE) for predictions.

    Args:
        predictor: An initialized FlowPredictor instance with loaded weights.
        data: NumPy array of trajectories [num_samples, 2, N, D].
        batch_size: Batch size for inference.
        method: The prediction method to use ("heun", "rk4", "euler").
        n_steps: Number of steps (only relevant for "euler").

    Returns:
        A tuple containing:
            - Total inference time (seconds).
            - Average inference time per sample (microseconds).
            - Mean Squared Error (MSE) between predicted x1 and true x1.
            Returns (0, 0, float('inf')) if data is empty.
    """
    total_samples = len(data)
    if total_samples == 0:
        logger.warning("Cannot calculate metrics: Input data is empty.")
        return 0.0, 0.0, float('inf')

    predictor.model.eval() # Ensure model is in eval mode

    # Create DataLoader for batching
    # Data is expected as np.ndarray [N, 2, N_nodes, D_feat]
    data_tensor = torch.tensor(data, dtype=torch.float32)
    dataset = TensorDataset(data_tensor)
    # DataLoader yields batches of shape [batch_size, 2, N_nodes, D_feat]
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_time_ns = 0  # Use nanoseconds for higher precision timing
    all_mse_values = []
    num_processed = 0

    logger.info(f"Starting inference metric calculation for {total_samples} samples...")
    logger.info(f"Method: {method.upper()}, Steps: {n_steps}, Batch Size: {batch_size}")

    with torch.no_grad():
        batch_iterator = tqdm(dataloader, desc=f"Inferencing ({method}, N={n_steps})", leave=False)
        for batch_idx, (batch_data,) in enumerate(batch_iterator):
            batch_data = batch_data.to(predictor.device) # Shape [B, 2, N, D]
            x0 = batch_data[:, 0]        # Shape [B, N, D]
            x1_true = batch_data[:, -1]  # Shape [B, N, D]
            current_batch_size = x0.shape[0]

            # Time the prediction step accurately
            start_time = time.perf_counter_ns()
            x1_pred = predictor.predict_final_state(x0, method=method, n_steps=n_steps)
            end_time = time.perf_counter_ns()
            total_time_ns += (end_time - start_time)

            # Calculate MSE for the batch
            # mse_loss reduction='none' gives loss per element. Mean over N, D dims.
            batch_mse_per_sample = F.mse_loss(x1_pred, x1_true, reduction='none').mean(dim=(1, 2)) # Shape [B]
            all_mse_values.extend(batch_mse_per_sample.cpu().numpy())
            num_processed += current_batch_size


    # Final calculations
    avg_time_ns_per_sample = total_time_ns / num_processed if num_processed > 0 else 0
    total_time_s = total_time_ns / 1_000_000_000
    avg_time_us_per_sample = avg_time_ns_per_sample / 1_000
    final_mse = np.mean(all_mse_values) if all_mse_values else float('inf')

    logger.info(f"Inference completed for {num_processed} samples.")
    logger.info(f"  Avg Time: {avg_time_us_per_sample:.2f} µs/sample")
    logger.info(f"  Total Time: {total_time_s:.3f} s")
    logger.info(f"  Final MSE: {final_mse:.6f}")

    return total_time_s, avg_time_us_per_sample, final_mse


# NOTE: This function is identical to the one in train_flow_matching.py
# Consider moving to a shared utils module if this project grows.
def load_trajectory_data(data_dir: str, file_pattern: str = DATA_FILE_PATTERN) -> np.ndarray:
    """
    Loads and combines trajectory data (x0, x1) from multiple NPZ files.

    Performs validation checks, pre-allocates memory, loads data efficiently,
    and shuffles the combined trajectories.

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
    local_logger = logging.getLogger(__name__ + ".load_trajectory_data") # Use sub-logger

    if not os.path.isdir(data_dir):
         local_logger.error(f"Data directory not found: {data_dir}")
         raise FileNotFoundError(f"Data directory not found: {data_dir}")

    full_pattern = os.path.join(data_dir, file_pattern)
    file_paths = sorted(glob.glob(full_pattern))
    local_logger.info(f"Searching for trajectory files matching '{full_pattern}'. Found {len(file_paths)} files.")

    if not file_paths:
        local_logger.error(f"No NPZ files found matching the pattern in {data_dir}.")
        raise ValueError(f"No NPZ files found matching '{file_pattern}' in {data_dir}")

    # --- First Pass: Metadata Scan ---
    total_samples = 0
    expected_shape_inner = None # (N, D)
    dtype = None
    valid_files = []
    local_logger.info("Scanning files to determine total size, shape, and dtype...")

    for i, file_path in enumerate(file_paths):
        try:
            with np.load(file_path, allow_pickle=False) as data:
                if 'x0' not in data or 'x1' not in data:
                    local_logger.warning(f"Skipping {os.path.basename(file_path)}: missing 'x0' or 'x1' key.")
                    continue
                # ... (rest of the validation checks are identical to training script) ...
                x0_data = data['x0']
                x1_data = data['x1']
                x0_shape = x0_data.shape
                x1_shape = x1_data.shape
                if x0_shape[0] != x1_shape[0]: continue # Count mismatch
                current_samples = x0_shape[0]
                if current_samples == 0: continue # Empty file
                if len(x0_shape) != 3 or len(x1_shape) != 3: continue # Dim mismatch
                current_inner_shape = x0_shape[1:]
                if x1_shape[1:] != current_inner_shape: continue # Inner shape mismatch
                current_dtype = x0_data.dtype
                if expected_shape_inner is None:
                    expected_shape_inner = current_inner_shape
                    dtype = current_dtype
                elif current_inner_shape != expected_shape_inner: continue
                elif current_dtype != dtype: continue

                total_samples += current_samples
                valid_files.append(file_path)

        except Exception as e:
            local_logger.error(f"Error reading metadata from {os.path.basename(file_path)}: {e}. Skipping.", exc_info=False) # Keep log cleaner
            continue

    if total_samples == 0 or expected_shape_inner is None:
        local_logger.error("Scan complete, but no valid trajectory data could be processed.")
        raise ValueError("No valid trajectory data found or processed.")
    local_logger.info(f"Scan complete. Total valid samples: {total_samples}. Using {len(valid_files)} files.")

    # --- Pre-allocation ---
    final_shape = (total_samples, 2) + expected_shape_inner
    local_logger.info(f"Attempting to allocate NumPy array with shape: {final_shape}, dtype: {dtype}")
    try:
        combined_trajectories = np.zeros(final_shape, dtype=dtype)
        local_logger.info("Memory allocation successful.")
    except MemoryError as e:
        local_logger.error(f"MemoryError: Failed to allocate array of shape {final_shape}.")
        raise e
    except Exception as e:
        local_logger.error(f"Unexpected error during array allocation: {e}")
        raise

    # --- Second Pass: Data Loading ---
    current_pos = 0
    local_logger.info(f"Loading data from {len(valid_files)} valid files...")
    # Use standard print for loading progress if tqdm is not desired here, or use logger.debug
    loading_bar = tqdm(valid_files, desc="Loading Trajectory Files", unit="file")
    for file_path in loading_bar:
        try:
            with np.load(file_path) as data:
                x0 = data['x0']
                x1 = data['x1']
                num_samples_in_file = x0.shape[0]
                start_idx = current_pos
                end_idx = current_pos + num_samples_in_file
                if end_idx > total_samples: # Truncate if needed
                    num_to_load = total_samples - start_idx
                    if num_to_load <=0 : continue
                    x0 = x0[:num_to_load]; x1 = x1[:num_to_load]
                    end_idx = total_samples
                combined_trajectories[start_idx:end_idx, 0] = x0
                combined_trajectories[start_idx:end_idx, 1] = x1
                current_pos = end_idx
        except Exception as e:
            local_logger.error(f"Error loading data from {os.path.basename(file_path)}: {e}. Skipping.", exc_info=False)

    if current_pos != total_samples:
        local_logger.warning(f"Loaded samples ({current_pos}) mismatch expected total ({total_samples}). Trimming.")
        combined_trajectories = combined_trajectories[:current_pos]
    if len(combined_trajectories) == 0:
        local_logger.error("Combined trajectory array is empty after loading.")
        raise ValueError("Failed to load any data into the combined array.")
    local_logger.info(f"Successfully loaded {len(combined_trajectories)} samples.")

    # --- Shuffle Data ---
    local_logger.info(f"Shuffling {len(combined_trajectories)} trajectories...")
    np.random.seed(42)
    shuffle_indices = np.random.permutation(len(combined_trajectories))
    combined_trajectories = combined_trajectories[shuffle_indices]
    local_logger.info("Data loading and shuffling complete.")

    return combined_trajectories


def setup_logging():
    """Configures basic logging for the prediction script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout # Log directly to stdout for prediction script
    )
    # Get the root logger used by basicConfig
    root_logger = logging.getLogger()
    # Optional: Set level for verbose libraries
    # logging.getLogger('matplotlib').setLevel(logging.WARNING)
    return root_logger


def main():
    """
    Main execution block for the prediction script.

    1. Sets up logging and configuration.
    2. Loads test/evaluation trajectory data.
    3. Initializes the FlowPredictor model.
    4. Loads pre-trained model weights.
    5. Iterates through specified sampling methods:
       - Calculates performance metrics (MSE, time).
       - Generates predictions for a visualization subset.
       - Creates and saves comparative PCA plots.
    6. Saves a summary of the comparative results.
    """
    # --- Setup ---
    logger = setup_logging() # Configure root logger

    # --- Configuration ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = DEFAULT_MODEL_DIR
    model_filename = BEST_MODEL_FILENAME # Use the best model from training
    model_path = os.path.join(model_dir, model_filename)
    data_dir = DEFAULT_DATA_DIR # Directory with data for evaluation
    results_dir = os.path.join(model_dir, "prediction_results") # Subdir for clarity
    viz_output_dir = os.path.join(results_dir, VISUALIZATION_SUBDIR)
    batch_size = DEFAULT_INFERENCE_BATCH_SIZE
    n_euler_steps = DEFAULT_EULER_STEPS
    methods_to_test = SUPPORTED_SAMPLING_METHODS

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(viz_output_dir, exist_ok=True)

    logger.info("--- Starting Prediction Script ---")
    logger.info(f"Using Device: {device}")
    logger.info(f"Model Path: {model_path}")
    logger.info(f"Data Directory: {data_dir}")
    logger.info(f"Results Directory: {results_dir}")
    logger.info(f"Visualization Directory: {viz_output_dir}")
    logger.info(f"Inference Batch Size: {batch_size}")
    logger.info(f"Euler Steps: {n_euler_steps}")
    logger.info(f"Methods to test: {methods_to_test}")


    # --- Load Data ---
    logger.info("Loading evaluation trajectory data...")
    try:
        trajectories = load_trajectory_data(data_dir=data_dir)
        logger.info(f"Loaded {len(trajectories)} trajectories with shape {trajectories.shape}")
        if len(trajectories) == 0: raise ValueError("Loaded data is empty.")
    except (FileNotFoundError, ValueError, MemoryError) as e:
        logger.error(f"Fatal Error: Failed to load evaluation data. {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.exception("An unexpected error occurred during data loading.")
        sys.exit(1)


    # --- Initialize Predictor and Load Weights ---
    logger.info("Initializing FlowPredictor...")
    try:
        predictor = FlowPredictor(device=device)
        logger.info(f"Loading model weights from {model_path}...")
        predictor.load_model_weights(model_path)
    except (FileNotFoundError, KeyError, RuntimeError, MemoryError) as e:
        logger.error(f"Fatal Error: Failed to initialize or load model. {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.exception("An unexpected error occurred during model setup.")
        sys.exit(1)


    # --- Prepare Visualization Subset ---
    num_total_samples = len(trajectories)
    n_viz_samples = min(MAX_VIZ_SAMPLES, num_total_samples)
    if n_viz_samples > 0:
        logger.info(f"Selecting {n_viz_samples} samples for visualization.")
        viz_data_np = trajectories[:n_viz_samples]
        # Convert only the necessary parts to tensors on the device
        viz_x0 = torch.tensor(viz_data_np[:, 0], dtype=torch.float32).to(device) # [viz_N, N, D]
        viz_x1_true = torch.tensor(viz_data_np[:, -1], dtype=torch.float32).to(device) # [viz_N, N, D]
    else:
        logger.warning("No samples available for visualization.")
        viz_x0, viz_x1_true = None, None


    # --- Evaluate Methods ---
    results = {}
    for method in methods_to_test:
        current_n_steps = n_euler_steps if method == "euler" else 1
        method_key = f"{method}_{current_n_steps}" if method == "euler" else method
        logger.info(f"\n--- Evaluating Method: {method.upper()} (Steps: {current_n_steps}) ---")

        # 1. Calculate Metrics on Full Dataset
        try:
            total_time_s, avg_time_us, mse = calculate_prediction_metrics(
                predictor=predictor,
                data=trajectories,
                batch_size=batch_size,
                method=method,
                n_steps=current_n_steps
            )
            results[method_key] = {
                "total_time_s": total_time_s,
                "avg_time_us": avg_time_us,
                "mse": mse,
                "n_steps": current_n_steps
            }
        except Exception as e:
            logger.error(f"Error calculating metrics for method {method}: {e}", exc_info=True)
            results[method_key] = {"error": str(e)} # Record error

        # 2. Generate Predictions and Visualize Subset (if possible)
        if n_viz_samples > 0 and 'error' not in results[method_key]:
            logger.info(f"Generating predictions for visualization subset ({method}, {current_n_steps} steps)...")
            try:
                with torch.no_grad():
                    viz_x1_pred = predictor.predict_final_state(viz_x0, method=method, n_steps=current_n_steps)

                # Create and save visualization plots
                predictor.visualize_predictions(
                    x0=viz_x0,
                    x1_true=viz_x1_true,
                    x1_pred=viz_x1_pred,
                    method=method,
                    n_steps=current_n_steps,
                    output_dir=viz_output_dir,
                    plots=["2d", "3d"]
                )
            except Exception as e:
                logger.error(f"Failed to generate visualization for method {method}: {e}", exc_info=True)


    # --- Save Comparative Results ---
    logger.info("\n--- Comparative Results Summary ---")
    results_path = os.path.join(results_dir, RESULTS_FILENAME)

    try:
        with open(results_path, "w") as f:
            f.write(f"Prediction Results\n")
            f.write(f"Model Path: {model_path}\n")
            f.write(f"Data Path: {data_dir}\n")
            f.write(f"Total samples evaluated: {num_total_samples}\n")
            f.write(f"Visualization samples: {n_viz_samples}\n\n")

            # Sort methods by speed (average time per sample in microseconds)
            sorted_methods = sorted(results.items(), key=lambda item: item[1].get('avg_time_us', float('inf')))

            summary_lines = []
            comparison_lines = []

            # Table Header
            header = f"{'Method':<15} | {'Steps':<7} | {'Avg Time/Sample (µs)':<20} | {'MSE':<12} | {'Samples/sec':<15}\n"
            separator = "-" * len(header) + "\n"
            comparison_lines.append(header)
            comparison_lines.append(separator)

            for method_key, r in sorted_methods:
                if 'error' in r:
                    summary = f"{method_key.upper()} Method Results:\n  ERROR: {r['error']}\n\n"
                    row = f"{method_key:<15} | {'N/A':<7} | {'ERROR':<20} | {'ERROR':<12} | {'ERROR':<15}\n"
                else:
                    avg_us = r['avg_time_us']
                    sps = 1_000_000 / avg_us if avg_us > 0 else 0
                    steps = r['n_steps']
                    mse = r['mse']
                    total_s = r['total_time_s']

                summary = (
                    f"{method_key.upper()} Method Results (Steps: {steps}):\n"
                    f"  Total inference time: {total_s:.3f} s\n"
                    f"  Average time per sample: {avg_us:.2f} µs\n"
                    f"  Mean Squared Error (MSE): {mse:.6f}\n"
                    f"  Samples per second (approx): {sps:.0f}\n\n"
                )
                row = f"{method_key:<15} | {steps:<7} | {avg_us:<20.2f} | {mse:<12.6f} | {sps:<15.0f}\n"

                summary_lines.append(summary)
                comparison_lines.append(row)

            # Write summaries first, then the table
            full_summary = "".join(summary_lines)
            full_comparison = "Method Comparison Table:\n" + separator + "".join(comparison_lines)

            logger.info("\n" + full_summary)
            f.write(full_summary)

            logger.info("\n" + full_comparison)
            f.write(full_comparison)

            logger.info(f"Results saved to {results_path}")

    except IOError as e:
        logger.error(f"Failed to write results file to {results_path}: {e}")
    except Exception as e:
        logger.exception("An unexpected error occurred while saving results.")


    logger.info("--- Prediction script finished ---")


if __name__ == "__main__":
    main()
