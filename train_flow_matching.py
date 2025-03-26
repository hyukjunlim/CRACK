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
from MPFlow import MPFlow, EnergyPredictionHead


class MPFlowEnergyPredictor:
    """
    Model that combines flow matching for trajectory generation with energy prediction.
    
    Key features:
    - Uses MPFlow for trajectory prediction (initial → final embedding)
    - Adds energy prediction head to predict scalar energy values
    - Can be trained end-to-end or with pretrained flow matching
    
    Args:
        embedding_dim: Dimension of the embedding
        flow_hidden_dims: Hidden dimensions for the flow matching model
        energy_hidden_dims: Hidden dimensions for the energy prediction head
        flow_lr: Learning rate for flow matching model
        energy_lr: Learning rate for energy prediction head
        use_attention: Whether to use attention in MPFlow
        device: Device to run the model on
    """
    def __init__(
        self, 
        embedding_dim, 
        flow_hidden_dims=[256, 512, 512, 256],
        energy_hidden_dims=[256, 128, 64],
        flow_lr=1e-4,
        energy_lr=1e-4,
        use_attention=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.embedding_dim = embedding_dim
        
        # Initialize the flow matching network
        self.flow_model = MPFlow(
            embedding_dim=embedding_dim,
            hidden_dims=flow_hidden_dims,
            use_attention=use_attention,
        ).to(device)
        
        # Initialize the energy prediction head
        self.energy_head = EnergyPredictionHead(
            embedding_dim=embedding_dim,
            hidden_dims=energy_hidden_dims
        ).to(device)
        
        # Setup optimizers with weight decay
        self.flow_optimizer = optim.AdamW(
            self.flow_model.parameters(), 
            lr=flow_lr, 
            betas=(0.9, 0.999),
            weight_decay=1e-5
        )
        
        self.energy_optimizer = optim.AdamW(
            self.energy_head.parameters(), 
            lr=energy_lr, 
            betas=(0.9, 0.999),
            weight_decay=1e-5
        )
        
        # Schedulers
        self.flow_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.flow_optimizer, 
            T_0=50,
            T_mult=2
        )
        
        self.energy_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.energy_optimizer, 
            T_0=50,
            T_mult=2
        )
    
    def train_step(self, x0, x1, energies, t, flow_weight=1.0, energy_weight=0.001):
        """
        Performs a joint training step for both flow matching and energy prediction.
        
        Args:
            x0: Initial embeddings of shape [batch_size, embedding_dim]
            x1: Final embeddings of shape [batch_size, embedding_dim]
            energies: Target energy values of shape [batch_size, 1]
            t: Time points of shape [batch_size, 1]
            flow_weight: Weight for flow matching loss
            energy_weight: Weight for energy prediction loss
            
        Returns:
            Total loss value and component losses
        """
        self.flow_model.train()
        self.energy_head.train()
        
        # Flow matching forward pass
        ut = x1 - x0
        xt = x0 + t * ut
        predicted_ut = self.flow_model(xt, t)
        
        # Flow matching loss
        mse_loss = F.mse_loss(predicted_ut, ut)
        direction_loss = (1 - torch.cosine_similarity(predicted_ut, ut, dim=1)).mean()
        flow_loss = mse_loss + direction_loss * 0.005
        
        # Generate final embeddings for energy prediction
        # Use detached x1 to avoid backpropagating through the ground truth
        # This ensures we're evaluating energy prediction on flow-generated embeddings
        with torch.no_grad():
            final_embeddings = self.sample_trajectory(x0, method="rk4")[:, -1]
        
        # Energy prediction
        predicted_energies = self.energy_head(final_embeddings)
        energy_loss = F.mse_loss(predicted_energies, energies)
        
        # Combined loss
        total_loss = flow_weight * flow_loss + energy_weight * energy_loss
        
        # Optimization step
        self.flow_optimizer.zero_grad()
        self.energy_optimizer.zero_grad()
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.flow_model.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self.energy_head.parameters(), max_norm=0.5)
        
        self.flow_optimizer.step()
        self.energy_optimizer.step()
        
        return total_loss.item(), flow_loss.item(), energy_loss.item()
    
    def sample_trajectory(self, x0, steps=2, method="rk4"):
        """
        Samples a trajectory from the flow matching model.
        
        Args:
            x0: Initial embeddings of shape [batch_size, embedding_dim]
            steps: Number of steps (currently only supports 2 for start/end)
            method: Integration method ("euler" or "rk4")
            
        Returns:
            Trajectory tensor of shape [batch_size, steps, embedding_dim]
        """
        self.flow_model.eval()
        batch_size = x0.shape[0]
        device = x0.device

        # For fixed step methods, just compute the final point
        x = x0.clone()
        
        with torch.no_grad():
            if method == "euler":
                # Single large step
                t = torch.ones((batch_size, 1), device=device)
                velocity = self.flow_model(x, t)
                x = x + velocity  # Single step from 0 to 1
                
            elif method == "rk4":
                # Single RK4 step from t=0 to t=1
                t = torch.zeros((batch_size, 1), device=device)
                k1 = self.flow_model(x, t)
                k2 = self.flow_model(x + 0.5 * k1, t + 0.5)
                k3 = self.flow_model(x + 0.5 * k2, t + 0.5)
                k4 = self.flow_model(x + k3, t + 1.0)
                
                x = x + (1.0 / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Return start and end points only
        trajectory = torch.stack([x0, x], dim=1)  # [batch, 2, dim]
        return trajectory
    
    def predict_energy(self, x0, method="rk4"):
        """
        Predicts energy values for given initial embeddings.
        
        Args:
            x0: Initial embeddings of shape [batch_size, embedding_dim]
            method: Integration method for generating final embeddings
            
        Returns:
            Predicted energy values of shape [batch_size, 1]
        """
        self.flow_model.eval()
        self.energy_head.eval()
        
        with torch.no_grad():
            # Generate final embeddings
            final_embeddings = self.sample_trajectory(x0, method=method)[:, -1]
            
            # Predict energies
            energies = self.energy_head(final_embeddings)
        
        return energies
    
    def save_model(self, path, save_flow=True, save_energy=True):
        """
        Saves the model state.
        
        Args:
            path: Base path for saving model files
            save_flow: Whether to save flow matching model
            save_energy: Whether to save energy prediction head
        """
        if save_flow:
            flow_path = f"{path}_flow.pt"
            torch.save({
                'model_state_dict': self.flow_model.state_dict(),
                'optimizer_state_dict': self.flow_optimizer.state_dict(),
            }, flow_path)
        
        if save_energy:
            energy_path = f"{path}_energy.pt"
            torch.save({
                'model_state_dict': self.energy_head.state_dict(),
                'optimizer_state_dict': self.energy_optimizer.state_dict(),
            }, energy_path)
    
    def load_model(self, path, load_flow=True, load_energy=True):
        """
        Loads the model state.
        
        Args:
            path: Base path for loading model files
            load_flow: Whether to load flow matching model
            load_energy: Whether to load energy prediction head
        """
        if load_flow:
            flow_path = f"{path}_flow.pt"
            if os.path.exists(flow_path):
                checkpoint = torch.load(flow_path)
                self.flow_model.load_state_dict(checkpoint['model_state_dict'])
                self.flow_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if load_energy:
            energy_path = f"{path}_energy.pt"
            if os.path.exists(energy_path):
                checkpoint = torch.load(energy_path)
                self.energy_head.load_state_dict(checkpoint['model_state_dict'])
                self.energy_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def visualize(self, trajectories, step=100, output_dir='flow_output', plots=["2d", "3d", "tsne"]):
        """
        Creates comprehensive visualizations of trajectories.
        Generates multiple plot types:
        - 2D PCA: Principal component projection
        - 3D PCA: 3D visualization of main components
        - t-SNE: Non-linear dimensionality reduction
        """
        # Convert trajectories to torch tensor
        trajectories = torch.tensor(trajectories, dtype=torch.float32, device=self.device)
        
        # Select samples to visualize
        viz_samples = min(8, len(trajectories))
        indices = np.arange(viz_samples)
        
        # Get ground truth trajectories for selected indices
        ground_truth = trajectories[indices]
        
        # Sample trajectories with fixed number of steps
        x0 = ground_truth[:, 0].to(self.device)
        x1 = ground_truth[:, -1].to(self.device)
        
        # Sample trajectories
        sampled_trajectories = self.sample_trajectory(x0, steps=2, method="rk4")
        
        # Create visualizations
        if "2d" in plots:
            self._create_2d_pca_plot(ground_truth, sampled_trajectories, output_dir)
        
        if "3d" in plots:
            self._create_3d_pca_plot(ground_truth, sampled_trajectories, output_dir)
        
        if "tsne" in plots:
            self._create_tsne_plot(ground_truth, sampled_trajectories, output_dir)

    def _create_2d_pca_plot(self, ground_truth, sampled_trajectories, output_dir):
        """Creates 2D PCA visualization comparing ground truth vs sampled trajectories."""
        # Combine for consistent PCA
        gt_flat = ground_truth.cpu().reshape(-1, ground_truth.shape[-1])
        sampled_flat = sampled_trajectories.cpu().reshape(-1, sampled_trajectories.shape[-1])
        
        # Apply PCA
        pca = PCA(n_components=2)
        gt_2d = pca.fit_transform(gt_flat.numpy())
        sampled_2d = pca.transform(sampled_flat.numpy())
        
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
            # Ground truth trajectory
            plt.plot(gt_2d[i, :, 0], gt_2d[i, :, 1], '--', color=colors[i], alpha=0.7, 
                    linewidth=2, label=f'GT {i+1}' if i == 0 else None)
            
            # Sampled trajectory
            plt.plot(sampled_2d[i, :, 0], sampled_2d[i, :, 1], '-', color=colors[i], 
                    alpha=0.9, linewidth=2, label=f'Sampled {i+1}' if i == 0 else None)
            
            # Mark points
            plt.scatter(gt_2d[i, 0, 0], gt_2d[i, 0, 1], color='blue', s=50, marker='o')
            plt.scatter(gt_2d[i, -1, 0], gt_2d[i, -1, 1], color='red', s=50, marker='o')
            plt.scatter(sampled_2d[i, 0, 0], sampled_2d[i, 0, 1], color='blue', s=50, marker='o')
            plt.scatter(sampled_2d[i, -1, 0], sampled_2d[i, -1, 1], color='black', s=25, marker='x')
        
        plt.title('Flow Trajectories - PCA Projection\n(Full GT vs Sampled Start/End)')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        plt.grid(True, alpha=0.3)
        plt.legend(['Ground Truth', 'Sampled', 'Start Points', 'End Points'])
        plt.savefig(os.path.join(output_dir, f"flow_trajectories_2d.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def _create_3d_pca_plot(self, ground_truth, sampled_trajectories, output_dir):
        """Creates 3D PCA visualization."""
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            # Apply PCA
            gt_flat = ground_truth.cpu().reshape(-1, ground_truth.shape[-1])
            sampled_flat = sampled_trajectories.cpu().reshape(-1, sampled_trajectories.shape[-1])
            
            pca = PCA(n_components=3)
            gt_3d = pca.fit_transform(gt_flat.numpy())
            sampled_3d = pca.transform(sampled_flat.numpy())
            
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
                # Plot trajectories and points
                ax.plot(gt_3d[i, :, 0], gt_3d[i, :, 1], gt_3d[i, :, 2], '--', 
                       color=colors[i], alpha=0.7, linewidth=2)
                ax.plot(sampled_3d[i, :, 0], sampled_3d[i, :, 1], sampled_3d[i, :, 2], 
                       '-', color=colors[i], alpha=0.9, linewidth=2)
                
                # Mark points
                ax.scatter(gt_3d[i, 0, 0], gt_3d[i, 0, 1], gt_3d[i, 0, 2], color='blue', s=50, marker='o')
                ax.scatter(gt_3d[i, -1, 0], gt_3d[i, -1, 1], gt_3d[i, -1, 2], color='red', s=50, marker='o')
                ax.scatter(sampled_3d[i, 0, 0], sampled_3d[i, 0, 1], sampled_3d[i, 0, 2], color='blue', s=50, marker='o')
                ax.scatter(sampled_3d[i, -1, 0], sampled_3d[i, -1, 1], sampled_3d[i, -1, 2], color='black', s=25, marker='x')
            
            ax.set_title('Flow Trajectories - 3D PCA Projection\n(Full GT vs Sampled Start/End)')
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
            
            # Custom legend
            from matplotlib.lines import Line2D
            custom_lines = [
                Line2D([0], [0], linestyle='--', color='gray', linewidth=2),
                Line2D([0], [0], linestyle='-', color='gray', linewidth=2),
                Line2D([0], [0], linestyle='none', marker='o', color='blue', markersize=10),
                Line2D([0], [0], linestyle='none', marker='o', color='red', markersize=10)
            ]
            ax.legend(custom_lines, ['Ground Truth', 'Sampled', 'Start Points', 'End Points'])
            
            plt.savefig(os.path.join(output_dir, f"flow_trajectories_3d.png"), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error creating 3D plot: {e}", flush=True)

    def _create_tsne_plot(self, ground_truth, sampled_trajectories, output_dir):
        """Creates t-SNE visualization for non-linear trajectory analysis."""
        try:
            from sklearn.manifold import TSNE
            
            # Prepare data
            gt_flat = ground_truth.cpu().reshape(-1, ground_truth.shape[-1]).numpy()
            sampled_flat = sampled_trajectories.cpu().reshape(-1, sampled_trajectories.shape[-1]).numpy()
            
            # Sample if data is too large
            max_points = 5000
            n_samples = ground_truth.shape[0]
            gt_timesteps = ground_truth.shape[1]
            sampled_timesteps = sampled_trajectories.shape[1]
            
            if gt_flat.shape[0] > max_points:
                n_trajectories = max_points // gt_timesteps
                selected_indices = np.random.choice(n_samples, n_trajectories, replace=False)
                gt_flat = gt_flat.reshape(n_samples, gt_timesteps, -1)[selected_indices].reshape(-1, ground_truth.shape[-1])
                sampled_flat = sampled_flat.reshape(n_samples, sampled_timesteps, -1)[selected_indices].reshape(-1, sampled_trajectories.shape[-1])
                n_samples = n_trajectories
            
            # Apply t-SNE
            tsne = TSNE(
                n_components=2, 
                perplexity=30, 
                n_iter=1000,
                init='pca',           # Use PCA initialization
                learning_rate='auto'  # Use automatic learning rate
            )
            combined = np.vstack([gt_flat, sampled_flat])
            combined_tsne = tsne.fit_transform(combined)
            
            # Split results
            gt_tsne = combined_tsne[:gt_flat.shape[0]].reshape(n_samples, gt_timesteps, 2)
            sampled_tsne = combined_tsne[gt_flat.shape[0]:].reshape(n_samples, sampled_timesteps, 2)
            
            # Create plot
            plt.figure(figsize=(12, 10))
            colors = plt.cm.tab20(np.linspace(0, 1, n_samples))
            
            for i in range(n_samples):
                plt.plot(gt_tsne[i, :, 0], gt_tsne[i, :, 1], '--', color=colors[i], alpha=0.7, linewidth=2)
                plt.plot(sampled_tsne[i, :, 0], sampled_tsne[i, :, 1], '-', color=colors[i], alpha=0.9, linewidth=2)
                
                plt.scatter(gt_tsne[i, 0, 0], gt_tsne[i, 0, 1], color='blue', s=50, marker='o')
                plt.scatter(gt_tsne[i, -1, 0], gt_tsne[i, -1, 1], color='red', s=50, marker='o')
                plt.scatter(sampled_tsne[i, 0, 0], sampled_tsne[i, 0, 1], color='blue', s=50, marker='o')
                plt.scatter(sampled_tsne[i, -1, 0], sampled_tsne[i, -1, 1], color='black', s=25, marker='x')
            
            plt.title('t-SNE Visualization\n(Full GT vs Sampled Start/End)')
            plt.grid(True, alpha=0.3)
            plt.legend(['Ground Truth', 'Sampled', 'Start Points', 'End Points'])
            plt.savefig(os.path.join(output_dir, f"tsne_visualization.png"), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error creating t-SNE plot: {e}", flush=True)


# Training function for the energy prediction model
def train_energy_predictor(
    trajectories, 
    energies,
    embedding_dim, 
    num_epochs=1550, 
    batch_size=32, 
    output_dir='energy_flow_output',
    validation_interval=10,
    flow_hidden_dims=[256, 512, 512, 256],
    energy_hidden_dims=[256, 128, 64],
    pretrained_flow_path='flow_output/flow_matching_model.pt',
    energy_weight=0.001
):
    """
    Modified training function that loads pretrained flow model and only trains energy predictor.
    
    Args:
        trajectories: Trajectory data of shape [n_samples, n_steps, embedding_dim]
        energies: Energy values of shape [n_samples, 1]
        embedding_dim: Dimension of the embeddings
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        output_dir: Directory for saving outputs
        validation_interval: Interval for validation
        flow_hidden_dims: Hidden dimensions for flow matching model
        energy_hidden_dims: Hidden dimensions for energy prediction head
        pretrained_flow_path: Path to pretrained flow matching model
        energy_weight: Weight for energy prediction loss in joint training
        
    Returns:
        Trained model and loss history
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Split data into train and validation sets
    n_samples = trajectories.shape[0]
    train_ratio = 0.8
    n_train = int(n_samples * train_ratio)
    
    train_trajectories = trajectories[:n_train]
    val_trajectories = trajectories[n_train:]
    
    train_energies = energies[:n_train]
    val_energies = energies[n_train:]
    
    # Initialize model
    model = MPFlowEnergyPredictor(
        embedding_dim=embedding_dim,
        flow_hidden_dims=flow_hidden_dims,
        energy_hidden_dims=energy_hidden_dims,
    )
    
    # Load pretrained flow model
    model.load_model(pretrained_flow_path, load_flow=True, load_energy=False)
    
    # Freeze flow model parameters
    for param in model.flow_model.parameters():
        param.requires_grad = False
    
    print(f"Loaded pretrained flow model from: {pretrained_flow_path}", flush=True)
    print(f"Energy head trainable parameters: {sum(p.numel() for p in model.energy_head.parameters())}", flush=True)
    
    # Training history
    energy_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience, patience_counter = 50, 0
    
    for epoch in tqdm(range(num_epochs)):
        epoch_energy_losses = []
        
        # Create batch indices
        batch_indices = np.random.permutation(len(train_trajectories))
        num_batches = len(batch_indices) // batch_size
        
        for i in range(num_batches):
            batch_idx = batch_indices[i * batch_size:(i + 1) * batch_size]
            
            # Get batch data
            batch_trajectories = torch.tensor(
                train_trajectories[batch_idx], 
                dtype=torch.float32,
                device=model.device
            )
            batch_energies = torch.tensor(
                train_energies[batch_idx], 
                dtype=torch.float32,
                device=model.device
            ).view(-1, 1)
            
            x0 = batch_trajectories[:, 0]
            
            # Only train energy head
            model.energy_optimizer.zero_grad()
            
            # Generate final embeddings using frozen flow model
            with torch.no_grad():
                final_embeddings = model.sample_trajectory(x0, method="rk4")[:, -1]
            
            # Energy prediction and loss
            predicted_energies = model.energy_head(final_embeddings)
            energy_loss = F.mse_loss(predicted_energies, batch_energies)
            
            energy_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.energy_head.parameters(), max_norm=0.5)
            model.energy_optimizer.step()
            
            epoch_energy_losses.append(energy_loss.item())
    
        # Update learning rate scheduler
        model.energy_scheduler.step()
        
        # Log training progress
        avg_energy_loss = sum(epoch_energy_losses) / len(epoch_energy_losses)
        energy_losses.append(avg_energy_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch + 1}/{num_epochs}, Energy Loss: {avg_energy_loss:.6f}", flush=True)
        
        # Validation
        if (epoch + 1) % validation_interval == 0 or epoch == num_epochs - 1:
            val_loss = validate_energy_predictor(model, val_trajectories, val_energies, batch_size)
            val_losses.append(val_loss)
            print(f"Validation Energy Loss: {val_loss:.6f}", flush=True)
            
            # Model checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_model(
                    os.path.join(output_dir, 'best_energy_model'),
                    save_flow=False,  # Don't save flow model since it's frozen
                    save_energy=True
                )
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs", flush=True)
                break
        
        # Regular checkpointing
        if (epoch + 1) % 100 == 0:
            model.save_model(
                os.path.join(output_dir, 'energy_model'),
                save_flow=False,
                save_energy=True
            )
    
    # Save final model
    model.save_model(
        os.path.join(output_dir, 'energy_model'),
        save_flow=False,
        save_energy=True
    )
    
    # Plot losses
    plot_training_curves([], energy_losses, val_losses, validation_interval, output_dir)
    
    # Load best model for return
    model.load_model(
        os.path.join(output_dir, 'best_energy_model'),
        load_flow=False,
        load_energy=True
    )
    
    return model, {
        'energy': energy_losses, 
        'val': val_losses
    }


def validate_energy_predictor(model, val_trajectories, val_energies, batch_size=32):
    """
    Validates the energy prediction model on validation data.
    
    Args:
        model: MPFlowEnergyPredictor model
        val_trajectories: Validation trajectories
        val_energies: Validation energy values
        batch_size: Batch size for validation
        
    Returns:
        Tuple of (average total loss, average flow loss, average energy loss)
    """
    model.flow_model.eval()
    model.energy_head.eval()
    energy_losses = []
    flow_losses = []
    
    with torch.no_grad():
        num_batches = len(val_trajectories) // batch_size + (1 if len(val_trajectories) % batch_size != 0 else 0)
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(val_trajectories))
            
            batch_trajectories = torch.tensor(
                val_trajectories[start_idx:end_idx], 
                dtype=torch.float32,
                device=model.device
            )
            batch_energies = torch.tensor(
                val_energies[start_idx:end_idx], 
                dtype=torch.float32,
                device=model.device
            ).view(-1, 1)
            
            x0 = batch_trajectories[:, 0]
            x1 = batch_trajectories[:, -1]
            
            # Calculate flow loss
            t = torch.ones((end_idx - start_idx, 1), device=model.device)
            ut = x1 - x0
            xt = x0 + t * ut
            predicted_ut = model.flow_model(xt, t)
            mse_loss = F.mse_loss(predicted_ut, ut)
            direction_loss = (1 - torch.cosine_similarity(predicted_ut, ut, dim=1)).mean()
            flow_loss = mse_loss + direction_loss * 0.005
            flow_losses.append(flow_loss.item())
            
            # Calculate energy loss
            final_embeddings = model.sample_trajectory(x0, method="rk4")[:, -1]
            predicted_energies = model.energy_head(final_embeddings)
            energy_loss = F.mse_loss(predicted_energies, batch_energies)
            energy_losses.append(energy_loss.item())
    
    avg_flow_loss = sum(flow_losses) / len(flow_losses)
    avg_energy_loss = sum(energy_losses) / len(energy_losses)
    avg_total_loss = avg_flow_loss + avg_energy_loss
    
    return avg_total_loss


def plot_training_curves(flow_losses, energy_losses, val_losses, validation_interval, output_dir):
    """
    Plots training and validation loss curves.
    
    Args:
        flow_losses: Flow matching losses
        energy_losses: Energy prediction losses
        val_losses: Validation losses
        validation_interval: Interval between validations
        output_dir: Output directory for plots
    """
    import matplotlib.pyplot as plt
    
    # Flow losses
    if flow_losses:
        plt.figure(figsize=(12, 6))
        plt.plot(flow_losses, 'b-', linewidth=2)
        plt.xlabel('Epoch'), plt.ylabel('Loss')
        plt.title('Flow Matching Loss')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'flow_loss_curve.png'))
        plt.close()
    
    # Energy losses
    plt.figure(figsize=(12, 6))
    plt.plot(energy_losses, 'g-', linewidth=2)
    plt.xlabel('Epoch'), plt.ylabel('Loss')
    plt.title('Energy Prediction Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'energy_loss_curve.png'))
    plt.close()
    
    # Validation losses
    if val_losses:
        plt.figure(figsize=(12, 6))
        val_epochs = list(range(validation_interval-1, len(energy_losses), validation_interval))
        if len(val_losses) > len(val_epochs):
            val_epochs.append(len(energy_losses) - 1)
        plt.plot(val_epochs[:len(val_losses)], val_losses, 'r-', linewidth=2)
        plt.xlabel('Epoch'), plt.ylabel('Loss')
        plt.title('Validation Loss')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'validation_loss_curve.png'))
        plt.close()


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
    
    data = np.load('logs/2316385/s2ef_predictions.npz')
    trajectories = data['latents'].reshape(-1, 2, 49 * 128)
    energies = data['energy']  # Pre-calculated energy values
    
    print(f"Loaded trajectories shape: {trajectories.shape}", flush=True)
    
    # Train the model
    embedding_dim = 128
    
    # Option 1: Train from scratch
    model, losses = train_energy_predictor(
        trajectories=trajectories,
        energies=energies,
        embedding_dim=embedding_dim,
        num_epochs=1550,
        batch_size=64,
        output_dir=output_dir,
    )
    
    print("Training completed and model saved!", flush=True)

