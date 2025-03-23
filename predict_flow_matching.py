import torch
import numpy as np
import time
import os
from train_flow_matching import FlowMatching
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

def load_and_prepare_data(data_path):
    """Load and prepare the data for inference"""
    data = np.load(data_path)
    trajectories = data['latents'].reshape(-1, 21, 128)
    return trajectories

def calculate_metrics(model, data, batch_size=32, device="cuda"):
    """
    Calculate inference time and MSE for the entire dataset
    Returns:
        - total_time: Total inference time
        - avg_time_per_sample: Average time per sample
        - mse: Mean squared error between predicted and actual final embeddings
    """
    model.model.eval()
    total_samples = len(data)
    
    # Convert to torch tensors
    data_tensor = torch.tensor(data, dtype=torch.float32)
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    total_time = 0
    all_mse = []
    
    print(f"Starting inference for {total_samples} samples...", flush=True)
    
    with torch.no_grad():
        for batch_idx, (batch,) in enumerate(dataloader):
            batch = batch.to(device)
            x0 = batch[:, 0]  # First embedding
            x1_true = batch[:, -1]  # True final embedding
            
            # Measure inference time
            start_time = time.time()
            trajectories = model.sample_trajectory(x0, method="rk4")
            x1_pred = trajectories[:, -1]  # Predicted final embedding
            batch_time = time.time() - start_time
            
            # Calculate MSE for this batch
            batch_mse = F.mse_loss(x1_pred, x1_true, reduction='none').mean(dim=1)
            all_mse.extend(batch_mse.cpu().numpy())
            
            total_time += batch_time
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {(batch_idx + 1) * batch_size}/{total_samples} samples", flush=True)
    
    avg_time_per_sample = total_time / total_samples
    mse = np.mean(all_mse)
    
    return total_time, avg_time_per_sample, mse

def main():
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "flow_output/exp2/best_flow_model.pt"
    data_path = "logs/2316385/s2ef_predictions.npz"
    embedding_dim = 128
    batch_size = 12182
    
    # Load data
    print("Loading data...", flush=True)
    trajectories = load_and_prepare_data(data_path)
    print(f"Loaded trajectories shape: {trajectories.shape}", flush=True)
    
    # Initialize model
    print("Initializing model...", flush=True)
    flow_model = FlowMatching(
        embedding_dim=embedding_dim,
        hidden_dims=[256, 512, 512, 256],
        device=device
    )
    
    # Load trained weights
    print("Loading model weights...", flush=True)
    flow_model.load_model(model_path)
    
    # Calculate metrics
    print("Starting inference...", flush=True)
    total_time, avg_time, mse = calculate_metrics(
        model=flow_model,
        data=trajectories,
        batch_size=batch_size,
        device=device
    )
    
    # Print results
    print("\nResults:", flush=True)
    print(f"Total inference time: {total_time:.2f} seconds", flush=True)
    print(f"Average time per sample: {avg_time*1000:.2f} milliseconds", flush=True)
    print(f"Mean Squared Error: {mse:.6f}", flush=True)
    
    # Save results to file
    results_path = "flow_output/inference_results.txt"
    with open(results_path, "w") as f:
        f.write(f"Total samples: {len(trajectories)}\n")
        f.write(f"Total inference time: {total_time:.2f} seconds\n")
        f.write(f"Average time per sample: {avg_time*1000:.2f} milliseconds\n")
        f.write(f"Mean Squared Error: {mse:.6f}\n")
    
    print(f"\nResults saved to {results_path}", flush=True)

if __name__ == "__main__":
    main()
