import torch
import numpy as np
import time
import os
from train_flow_matching import FlowMatching
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

def calculate_metrics(model, data, batch_size=32, device="cuda", method="dopri5"):
    """
    Calculate inference time and MSE with microsecond precision
    """
    model.model.eval()
    total_samples = len(data)
    
    data_tensor = torch.tensor(data, dtype=torch.float32)
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    total_time_us = 0  # Track time in microseconds
    all_mse = []
    
    print(f"Starting inference with {method} method for {total_samples} samples...", flush=True)
    
    with torch.no_grad():
        for batch_idx, (batch,) in enumerate(dataloader):
            batch = batch.to(device)
            x0 = batch[:, 0]
            x1_true = batch[:, -1]
            
            # Measure time in microseconds
            start_time = time.perf_counter()
            trajectories = model.sample_trajectory(x0, method=method)
            x1_pred = trajectories[:, -1]
            batch_time = (time.perf_counter() - start_time) * 1_000_000  # Convert to microseconds
            
            batch_mse = F.mse_loss(x1_pred, x1_true, reduction='none').mean(dim=(1, 2))
            all_mse.extend(batch_mse.cpu().numpy())
            
            total_time_us += batch_time
            
            if (batch_idx + 1) % 10 == 0:
                avg_mse = np.mean(all_mse)
                time_ms = total_time_us / 1000  # Convert to milliseconds for display
                print(f"Processed {(batch_idx + 1) * batch_size}/{total_samples} samples. "
                      f"Current MSE: {avg_mse:.6f}, "
                      f"Time: {time_ms:.2f}ms", flush=True)
    
    avg_time_us = total_time_us / total_samples
    mse = np.mean(all_mse)
    
    return total_time_us, avg_time_us, mse


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


def main():
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "flow_output/exp1/best_flow_model.pt"  # Updated path
    embedding_dim = 128
    batch_size = 12182
    methods = ["rk4", "heun", "euler"]  # Test all methods
    
    # Load data
    print("Loading data...", flush=True)
    trajectories = load_trajectory_data()
    print(f"Loaded trajectories shape: {trajectories.shape}", flush=True)
    
    # Initialize model
    print("Initializing model...", flush=True)
    flow_model = FlowMatching(
        embedding_dim=embedding_dim,
        seq_length=49,  # Added sequence length
        hidden_dims=[256, 512, 512, 256],
        device=device
    )
    
    # Load trained weights
    print("Loading model weights...", flush=True)
    flow_model.load_model(model_path)
    
    # Results dictionary to store metrics for each method
    results = {}
    
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
            "total_time": total_time,
            "avg_time": avg_time,
            "mse": mse
        }
    
    # Print and save comparative results
    print("\nComparative Results:", flush=True)
    results_path = "flow_output/exp1/inference_results.txt"
    
    with open(results_path, "w") as f:
        f.write(f"Total samples: {len(trajectories)}\n\n")
        
        # Sort methods by speed
        sorted_methods = sorted(results.items(), key=lambda x: x[1]['avg_time'])
        
        for method, r in sorted_methods:
            # Convert times to appropriate units
            total_ms = r['total_time'] / 1000  # Convert µs to ms
            avg_us = r['avg_time']  # Keep in µs for precision
            
            summary = (
                f"{method.upper()} Method Results:\n"
                f"Total inference time: {total_ms:.2f} ms\n"
                f"Average time per sample: {avg_us:.2f} µs\n"
                f"Mean Squared Error: {r['mse']:.6f}\n"
                f"Samples per second: {1_000_000/avg_us:.0f}\n\n"
            )
            print(summary, flush=True)
            f.write(summary)
            
        # Add comparison table
        comparison = "\nMethod Comparison Table:\n"
        comparison += "-" * 65 + "\n"
        comparison += f"{'Method':<10} | {'Time/Sample (µs)':<15} | {'MSE':<12} | {'Samples/sec':<10}\n"
        comparison += "-" * 65 + "\n"
        
        for method, r in sorted_methods:
            sps = 1_000_000/r['avg_time']
            comparison += f"{method:<10} | {r['avg_time']:15.2f} | {r['mse']:<12.6f} | {sps:10.0f}\n"
        
        print(comparison, flush=True)
        f.write(comparison)
    
    print(f"Results saved to {results_path}", flush=True)

if __name__ == "__main__":
    main()
