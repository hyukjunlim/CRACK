import torch
import numpy as np
import time
import os
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import logging
import sys
import glob
import torch.nn as nn
import torch.optim as optim
from nets.equiformer_v2.MPFlow import EquivariantMPFlow
from nets.equiformer_v2.so3 import SO3_Grid
from nets.equiformer_v2.module_list import ModuleListInfo
from tqdm import tqdm

class FlowMatching:
    """
    A class to manage flow matching on message passing trajectories.
    Implements training, sampling, and visualization of flow-based trajectories.

    Key features:
    - Supports multiple sampling methods (heun, rk4, euler) - NOTE: dopri5 removed for prediction script

    Args:
        lr: Learning rate (only needed for optimizer init, can be default for inference)
        device: Device to run the model on
    """
    def __init__(
        self,
        lr=1e-4, # Keep default for potential loading logic, though optimizer isn't used
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

        # Optimizer setup is not strictly needed for inference, but load_model expects it
        # We will modify load_model to only load the model state_dict
        # self.optimizer = optim.AdamW(
        #     self.model.parameters(),
        #     lr=lr,
        #     betas=(0.9, 0.999),
        #     weight_decay=1e-5
        # )

    def sample_trajectory(self, x0, method="heun"):
        """
        Samples a trajectory using ODE solver methods based on train_flow_matching.py.

        Args:
            x0: Starting point tensor of shape [batch_size, 49, 128]
            method: Sampling method ("heun", "rk4", "euler")
        """
        self.model.eval()
        batch_size = x0.shape[0]
        device = x0.device

        x = x0.clone()
        with torch.no_grad():
            if method == "heun":
                # Time tensor should be [batch_size, 1]
                t = torch.zeros((batch_size, 1), device=device)
                # Model expects input shape [B, 49, D] and time [B, 1]
                k1 = self.model(x, t) # k1 has shape [B, 49, D]
                x_pred = x + k1
                # Time t+1 should also be [batch_size, 1]
                t_next = torch.ones((batch_size, 1), device=device)
                k2 = self.model(x_pred, t_next)
                x = x + 0.5 * (k1 + k2) # Update uses shapes [B, 49, D]

            elif method == "rk4":
                # Time tensors should be [batch_size, 1]
                t = torch.zeros((batch_size, 1), device=device)
                t_half = torch.full((batch_size, 1), 0.5, device=device)
                t_one = torch.ones((batch_size, 1), device=device)
                # Model calls expect x=[B, 49, D], t=[B, 1] -> output=[B, 49, D]
                k1 = self.model(x, t)
                k2 = self.model(x + 0.5 * k1, t_half)
                k3 = self.model(x + 0.5 * k2, t_half)
                k4 = self.model(x + k3, t_one)
                x = x + (1.0 / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

            elif method == "euler":
                # Time tensor should be [batch_size, 1]
                t = torch.zeros((batch_size, 1), device=device)
                velocity = self.model(x, t) # velocity shape [B, 49, D]
                x = x + velocity # Update shape [B, 49, D]

            else:
                raise ValueError(f"Unsupported sampling method: {method}")

        # Return shape [batch_size, 2, 49, 128] matching the input data format
        trajectory = torch.stack([x0, x], dim=1)
        return trajectory

    def load_model(self, path):
        """Load the model state only."""
        # Using basic logger if none configured externally
        logger = logging.getLogger()
        logger.info(f"Loading model state ONLY from checkpoint: {path}")
        try:
            # Load checkpoint onto the correct device directly
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Model state loaded successfully.")
        except FileNotFoundError:
            logger.error(f"Error: Checkpoint file not found at {path}")
            raise
        except KeyError:
            logger.error(f"Error: 'model_state_dict' key not found in the checkpoint file {path}.")
            raise
        except Exception as e:
            logger.error(f"An error occurred while loading the model state: {e}")
            raise

def calculate_metrics(model, data, batch_size=32, device="cuda", method="heun"):
    """
    Calculate inference time and MSE with microsecond precision.
    Adapted for the new FlowMatching model and data format.
    """
    model.model.eval() # Ensure the inner model is in eval mode
    total_samples = len(data)

    # Data is expected to be numpy array (N, 2, 49, 128)
    data_tensor = torch.tensor(data, dtype=torch.float32)
    dataset = TensorDataset(data_tensor)
    # Note: DataLoader will yield batches of shape [batch_size, 2, 49, 128]
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_time_us = 0  # Track time in microseconds
    all_mse = []

    print(f"Starting inference with {method} method for {total_samples} samples...", flush=True)

    with torch.no_grad():
        for batch_idx, (batch,) in enumerate(tqdm(dataloader, desc=f"Inferencing ({method})")): # Add tqdm progress bar
            batch = batch.to(device) # Shape [B, 2, 49, 128]
            x0 = batch[:, 0]        # Shape [B, 49, 128]
            x1_true = batch[:, -1]  # Shape [B, 49, 128]

            # Measure time in microseconds
            start_time = time.perf_counter()
            # sample_trajectory expects x0 [B, 49, 128]
            # and returns trajectory [B, 2, 49, 128]
            trajectories = model.sample_trajectory(x0, method=method)
            x1_pred = trajectories[:, -1] # Shape [B, 49, 128]
            batch_time = (time.perf_counter() - start_time) * 1_000_000  # Convert to microseconds

            # Calculate MSE over batch, average over seq_len and embedding_dim
            batch_mse = F.mse_loss(x1_pred, x1_true, reduction='none').mean(dim=(1, 2)) # Mean over (49, 128) -> Shape [B]
            all_mse.extend(batch_mse.cpu().numpy())

            total_time_us += batch_time

            # Log progress less frequently to avoid console spam with tqdm
            # if (batch_idx + 1) % 10 == 0:
            #     avg_mse = np.mean(all_mse)
            #     time_ms = total_time_us / 1000  # Convert to milliseconds for display
            #     print(f"Processed {(batch_idx + 1) * batch_size}/{total_samples} samples. "
            #           f"Current Avg MSE: {avg_mse:.6f}, "
            #           f"Total Time: {time_ms:.2f}ms", flush=True)

    avg_time_us = total_time_us / total_samples
    mse = np.mean(all_mse)
    print(f"Inference with {method} completed. Avg Time: {avg_time_us:.2f} µs/sample, Final MSE: {mse:.6f}", flush=True)

    return total_time_us, avg_time_us, mse

def load_trajectory_data():
    """
    Loads and combines trajectory data from multiple NPZ files more memory-efficiently.
    Pre-allocates the final array and loads data directly into slices.
    Shuffles the combined trajectories along the batch dimension after loading.
    Uses data from 'mpflow_data' directory.

    Returns:
        Combined and shuffled trajectories array of shape (n_trajectories, 2, 49, 128)
    """
    # Using basic logger if none configured externally
    logger = logging.getLogger(__name__) # Use module-specific logger
    logger.setLevel(logging.INFO) # Ensure logs are shown
    if not logger.handlers: # Add handler if none exists (e.g., when run directly)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)


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

                # Check shape consistency (expecting N, 49, 128)
                expected_inner_shape = (49, 128)
                if len(x0_shape) != 3 or x0_shape[1:] != expected_inner_shape:
                     logger.warning(f"Skipping {file_path}: unexpected x0 shape {x0_shape}. Expected (N, {expected_inner_shape[0]}, {expected_inner_shape[1]}).")
                     continue
                if len(x1_shape) != 3 or x1_shape[1:] != expected_inner_shape:
                     logger.warning(f"Skipping {file_path}: unexpected x1 shape {x1_shape}. Expected (N, {expected_inner_shape[0]}, {expected_inner_shape[1]}).")
                     continue

                # Store shape and dtype from the first valid file
                # The final array will store (x0, x1) pair, so shape is (2, 49, 128)
                current_data_shape = (2,) + x0_shape[1:]
                current_dtype = data['x0'].dtype

                if sample_shape is None:
                    sample_shape = current_data_shape # Should be (2, 49, 128)
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

            # Log progress occasionally during scan
            # if (i + 1) % 50 == 0:
            #      logger.info(f"Scanned {i+1}/{len(file_paths)} files. Current valid samples: {total_samples}")

        except Exception as e:
            logger.error(f"Error reading metadata from {file_path}: {e}. Skipping.")
            continue # Skip file on error

    if total_samples == 0 or sample_shape is None:
        logger.error("No valid trajectory data could be processed from any files.")
        raise ValueError("No valid trajectory data found or could be processed.")

    logger.info(f"Scan complete. Total valid samples: {total_samples}, Sample shape per trajectory: {sample_shape}, Dtype: {dtype}")

    # Pre-allocate the array
    final_shape = (total_samples,) + sample_shape # (N, 2, 49, 128)
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
    # Use basic print for progress here if tqdm isn't imported/used in predict script
    for i, file_path in enumerate(valid_files):
         try:
             with np.load(file_path) as data:
                 x0 = data['x0'] # [N, 49, 128]
                 x1 = data['x1'] # [N, 49, 128]
                 num_samples_in_file = x0.shape[0]

                 if current_pos + num_samples_in_file > total_samples:
                      logger.warning(f"Data from {file_path} exceeds allocated space. Truncating load.")
                      num_samples_in_file = total_samples - current_pos
                      if num_samples_in_file <= 0: continue

                 # Stack and place into the correct slice
                 combined_trajectories[current_pos : current_pos + num_samples_in_file, 0] = x0[:num_samples_in_file]
                 combined_trajectories[current_pos : current_pos + num_samples_in_file, 1] = x1[:num_samples_in_file]

                 current_pos += num_samples_in_file

                 # Print progress update
                 if (i + 1) % 10 == 0 or i == len(valid_files) - 1:
                     print(f"Loaded {i+1}/{len(valid_files)} files ({num_samples_in_file} samples). Current position: {current_pos}/{total_samples}", flush=True)


         except Exception as e:
             logger.error(f"Error loading data from {file_path}: {e}. Skipping file, data may be incomplete.")

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

def main():
    # Basic logger setup for main function execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # --- IMPORTANT: Update path to match the output of train_flow_matching.py ---
    model_path = "flow_output/exp1/best_flow_model.pt" # Assuming exp1 was used
    output_dir = os.path.dirname(model_path) # Use the same dir for results
    os.makedirs(output_dir, exist_ok=True) # Ensure output dir exists

    # Batch size for inference (adjust based on GPU memory)
    # The value 12182 from the original script might be too large depending on the model size.
    # Start with a smaller value and increase if possible.
    batch_size = 10000 # Reduced batch size for potentially larger model

    # Methods available in the copied sample_trajectory
    methods = ["rk4", "heun", "euler"]

    # Load data using the new function
    print("Loading data using the updated function...", flush=True)
    trajectories = load_trajectory_data() # Shape (N, 2, 49, 128)
    print(f"Loaded trajectories shape: {trajectories.shape}", flush=True)

    # Initialize the updated FlowMatching model
    print("Initializing updated FlowMatching model...", flush=True)
    # We don't need to pass embedding_dim or seq_length anymore
    flow_model = FlowMatching(device=device)
    print(f"Model parameters: {sum(p.numel() for p in flow_model.model.parameters())}")


    # Load trained weights using the updated load_model
    print(f"Loading model weights from {model_path}...", flush=True)
    try:
        flow_model.load_model(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}", flush=True)
        sys.exit(1) # Exit if model loading fails

    # Results dictionary to store metrics for each method
    results = {}
    # Add tqdm import for calculate_metrics progress bar
    from tqdm import tqdm

    # Test each sampling method
    for method in methods:
        print(f"\nTesting {method} sampling method...", flush=True)
        total_time, avg_time, mse = calculate_metrics(
            model=flow_model,
            data=trajectories,
            batch_size=batch_size,
            device=device,
            method=method
        )

        results[method] = {
            "total_time": total_time, # in microseconds
            "avg_time": avg_time,     # in microseconds
            "mse": mse
        }

    # Print and save comparative results
    print("\nComparative Results:", flush=True)
    results_path = os.path.join(output_dir, "inference_results.txt") # Save in the model's output dir

    with open(results_path, "w") as f:
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Total samples evaluated: {len(trajectories)}\n\n")

        # Sort methods by speed (average time per sample in microseconds)
        sorted_methods = sorted(results.items(), key=lambda x: x[1]['avg_time'])

        for method, r in sorted_methods:
            # Convert times to appropriate units for display
            total_s = r['total_time'] / 1_000_000  # Convert µs to seconds
            avg_us = r['avg_time']                 # Keep in µs for precision
            sps = 1_000_000 / avg_us if avg_us > 0 else 0 # Samples per second

            summary = (
                f"{method.upper()} Method Results:\n"
                f"Total inference time: {total_s:.3f} s\n"
                f"Average time per sample: {avg_us:.2f} µs\n"
                f"Mean Squared Error (MSE): {r['mse']:.6f}\n"
                f"Samples per second (approx): {sps:.0f}\n\n"
            )
            print(summary, flush=True)
            f.write(summary)

        # Add comparison table
        comparison = "\nMethod Comparison Table:\n"
        comparison += "-" * 65 + "\n"
        comparison += f"{'Method':<10} | {'Time/Sample (µs)':<15} | {'MSE':<12} | {'Samples/sec':<15}\n"
        comparison += "-" * 65 + "\n"

        for method, r in sorted_methods:
            sps = 1_000_000 / r['avg_time'] if r['avg_time'] > 0 else 0
            comparison += f"{method:<10} | {r['avg_time']:<15.2f} | {r['mse']:<12.6f} | {sps:<15.0f}\n"

        print(comparison, flush=True)
        f.write(comparison)

    print(f"Results saved to {results_path}", flush=True)

if __name__ == "__main__":
    main()
