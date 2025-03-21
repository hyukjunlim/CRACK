import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import os

def visualize_embeddings(latents, energy, method='pca', perplexity=50, n_components=2, figsize=(12, 10), highlight_range=False, by_layers=False):
    """
    Visualize high-dimensional embeddings with energy values as colors.
    
    Parameters:
    -----------
    latents : numpy.ndarray
        Embedding vectors of shape (n_samples, embedding_dimension) or (n_samples, n_layers, embedding_dimension)
    energy : numpy.ndarray
        Energy values of shape (n_samples, 1)
    method : str
        Dimensionality reduction method ('pca' or 'tsne')
    perplexity : int
        Perplexity parameter for t-SNE (only used if method='tsne')
    n_components : int
        Number of dimensions to reduce to (2 or 3)
    figsize : tuple
        Figure size for the plots
    highlight_range : bool
        If True, highlights points with energy between 1.5 and 2
    by_layers : bool
        If True, create subplot for each layer in 3D input
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure containing the plots
    """
    # Flatten energy if it's a 2D array
    if energy.ndim > 1:
        energy = energy.ravel()
    
    if by_layers and latents.ndim == 3:
        n_layers = latents.shape[1]
        n_cols = 5
        n_rows = 5
        fig = plt.figure(figsize=(40, 30))
        
        for layer in range(n_layers):
            ax = plt.subplot(n_rows, n_cols, layer + 1)
            latents_layer = latents[:, layer, :]
            
            # Dimensionality reduction
            if method.lower() == 'pca':
                reducer = PCA(n_components=n_components)
                title_prefix = 'PCA'
            else:
                reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
                title_prefix = 't-SNE'
            
            reduced_data = reducer.fit_transform(latents_layer)
            highlight_mask = (energy >= 1.5) & (energy <= 2) if highlight_range else np.ones_like(energy, dtype=bool)
            
            if highlight_range:
                ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c='lightgray', alpha=0.5, s=50, label='Other samples')
                im = ax.scatter(reduced_data[highlight_mask, 0], reduced_data[highlight_mask, 1],
                              c=energy[highlight_mask], cmap='viridis', alpha=0.8, s=50, label='1.5 ≤ Energy ≤ 2')
            else:
                im = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=energy, cmap='viridis', alpha=0.8, s=50)
            
            plt.colorbar(im, ax=ax, label='Energy')
            ax.set_title(f'Layer {layer}: {title_prefix} Visualization')
            ax.set_xlabel(f'{title_prefix} Component 1')
            ax.set_ylabel(f'{title_prefix} Component 2')
            if highlight_range:
                ax.legend()
        
        plt.tight_layout()
        return fig
    
    # Create a figure with subplots
    fig = plt.figure(figsize=figsize)
    
    # Dimensionality reduction
    if method.lower() == 'pca':
        reducer = PCA(n_components=n_components)
        title_prefix = 'PCA'
    else:
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        title_prefix = 't-SNE'
    
    reduced_data = reducer.fit_transform(latents)
    
    # Set up colormap
    cmap = plt.cm.viridis
    
    # Create a mask for samples with energy between 1.5 and 2
    highlight_mask = (energy >= 1.5) & (energy <= 2) if highlight_range else np.ones_like(energy, dtype=bool)
    
    # 2D plot
    if n_components == 2:
        if highlight_range:
            # Plot all points in gray first
            plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c='lightgray', alpha=0.5, s=50, label='Other samples')
            # Plot highlighted points with energy colormap
            plt.scatter(reduced_data[highlight_mask, 0], reduced_data[highlight_mask, 1], 
                       c=energy[highlight_mask], cmap=cmap, alpha=0.8, s=50, label='1.5 ≤ Energy ≤ 2')
        else:
            plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=energy, cmap=cmap, alpha=0.8, s=50)
        plt.colorbar(label='Energy')
        plt.title(f'{title_prefix} Visualization of Catalyst Embeddings')
        plt.xlabel(f'{title_prefix} Component 1')
        plt.ylabel(f'{title_prefix} Component 2')
        plt.legend()
    
    # 3D plot
    elif n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        if highlight_range:
            # Plot all points in gray first
            ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], 
                      c='lightgray', alpha=0.5, s=50, label='Other samples')
            # Plot highlighted points with energy colormap
            scatter = ax.scatter(reduced_data[highlight_mask, 0], reduced_data[highlight_mask, 1], 
                               reduced_data[highlight_mask, 2], c=energy[highlight_mask], 
                               cmap=cmap, alpha=0.8, s=50, label='1.5 ≤ Energy ≤ 2')
        else:
            scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], 
                               c=energy, cmap=cmap, alpha=0.8, s=50)
        fig.colorbar(scatter, label='Energy')
        ax.set_title(f'{title_prefix} Visualization of Catalyst Embeddings')
        ax.set_xlabel(f'{title_prefix} Component 1')
        ax.set_ylabel(f'{title_prefix} Component 2')
        ax.set_zlabel(f'{title_prefix} Component 3')
        ax.legend()
    
    plt.tight_layout()
    return fig

def create_multiple_visualizations(latents, energy, highlight_range=False, by_layers=False):
    """
    Create multiple visualizations with different dimensionality reduction techniques.
    
    Parameters:
    -----------
    latents : numpy.ndarray
        Embedding vectors of shape (n_samples, embedding_dimension)
    energy : numpy.ndarray
        Energy values of shape (n_samples, 1)
    highlight_range : bool
        If True, highlights points with energy between 1.5 and 2
    by_layers : bool
        If True, create visualizations for each layer in 3D input
    """
    if by_layers and latents.ndim == 3:
        n_layers = latents.shape[1]
        n_cols = 5
        n_rows = 5
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 25))
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        
        if energy.ndim > 1:
            energy = energy.ravel()
        
        highlight_mask = (energy >= 1.5) & (energy <= 2) if highlight_range else np.ones_like(energy, dtype=bool)
        
        for layer in range(n_layers):
            row = layer // n_cols
            col = layer % n_cols
            latents_layer = latents[:, layer, :]
            
            # PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(latents_layer)
            
            if highlight_range:
                axes[row, col].scatter(pca_result[:, 0], pca_result[:, 1], c='lightgray', alpha=0.5, s=50)
                im1 = axes[row, col].scatter(pca_result[highlight_mask, 0], pca_result[highlight_mask, 1],
                                           c=energy[highlight_mask], cmap='viridis', alpha=0.8, s=50)
            else:
                im1 = axes[row, col].scatter(pca_result[:, 0], pca_result[:, 1], c=energy, cmap='viridis', alpha=0.8, s=50)
            
            axes[row, col].set_title(f'Layer {layer}')
            fig.colorbar(im1, ax=axes[row, col])
        
        # Hide empty subplots
        for i in range(layer + 1, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
    
    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Flatten energy if it's a 2D array
    if energy.ndim > 1:
        energy = energy.ravel()
    
    # Create a mask for samples with energy between 1.5 and 2
    highlight_mask = (energy >= 1.5) & (energy <= 2) if highlight_range else np.ones_like(energy, dtype=bool)
    
    # 1. PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(latents)
    
    if highlight_range:
        # Plot all points in gray first
        axes[0, 0].scatter(pca_result[:, 0], pca_result[:, 1], c='lightgray', alpha=0.5, s=50, label='Other samples')
        # Plot highlighted points with energy colormap
        im1 = axes[0, 0].scatter(pca_result[highlight_mask, 0], pca_result[highlight_mask, 1], 
                                c=energy[highlight_mask], cmap='viridis', alpha=0.8, s=50, label='1.5 ≤ Energy ≤ 2')
    else:
        im1 = axes[0, 0].scatter(pca_result[:, 0], pca_result[:, 1], c=energy, cmap='viridis', alpha=0.8, s=50)
    axes[0, 0].set_title('PCA Visualization')
    axes[0, 0].set_xlabel('PCA Component 1')
    axes[0, 0].set_ylabel('PCA Component 2')
    axes[0, 0].legend()
    fig.colorbar(im1, ax=axes[0, 0], label='Energy')
    
    # 2. t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(latents)
    
    if highlight_range:
        # Plot all points in gray first
        axes[0, 1].scatter(tsne_result[:, 0], tsne_result[:, 1], c='lightgray', alpha=0.5, s=50, label='Other samples')
        # Plot highlighted points with energy colormap
        im2 = axes[0, 1].scatter(tsne_result[highlight_mask, 0], tsne_result[highlight_mask, 1], 
                                c=energy[highlight_mask], cmap='viridis', alpha=0.8, s=50, label='1.5 ≤ Energy ≤ 2')
    else:
        im2 = axes[0, 1].scatter(tsne_result[:, 0], tsne_result[:, 1], c=energy, cmap='viridis', alpha=0.8, s=50)
    axes[0, 1].set_title('t-SNE Visualization (Perplexity=30)')
    axes[0, 1].set_xlabel('t-SNE Component 1')
    axes[0, 1].set_ylabel('t-SNE Component 2')
    axes[0, 1].legend()
    fig.colorbar(im2, ax=axes[0, 1], label='Energy')
    
    # 3. t-SNE with higher perplexity
    tsne_high = TSNE(n_components=2, perplexity=50, random_state=42)
    tsne_high_result = tsne_high.fit_transform(latents)
    
    if highlight_range:
        # Plot all points in gray first
        axes[1, 0].scatter(tsne_high_result[:, 0], tsne_high_result[:, 1], c='lightgray', alpha=0.5, s=50, label='Other samples')
        # Plot highlighted points with energy colormap
        im3 = axes[1, 0].scatter(tsne_high_result[highlight_mask, 0], tsne_high_result[highlight_mask, 1],
                                c=energy[highlight_mask], cmap='viridis', alpha=0.8, s=50, label='1.5 ≤ Energy ≤ 2')
    else:
        im3 = axes[1, 0].scatter(tsne_high_result[:, 0], tsne_high_result[:, 1], c=energy, cmap='viridis', alpha=0.8, s=50)
    axes[1, 0].set_title('t-SNE Visualization (Perplexity=50)')
    axes[1, 0].set_xlabel('t-SNE Component 1')
    axes[1, 0].set_ylabel('t-SNE Component 2')
    axes[1, 0].legend()
    fig.colorbar(im3, ax=axes[1, 0], label='Energy')
    
    # 4. Energy distribution
    sns.histplot(energy, kde=True, ax=axes[1, 1], color='teal')
    axes[1, 1].set_title('Energy Distribution')
    axes[1, 1].set_xlabel('Energy')
    axes[1, 1].set_ylabel('Count')
    
    plt.suptitle('Catalyst Embeddings Analysis', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    return fig

def analyze_embedding_clusters(latents, energy, n_clusters=5, by_layers=False):
    """
    Analyze embedding clusters and their relation to energy.
    
    Parameters:
    -----------
    latents : numpy.ndarray
        Embedding vectors of shape (n_samples, embedding_dimension)
    energy : numpy.ndarray
        Energy values of shape (n_samples, 1)
    n_clusters : int
        Number of clusters to form
    """
    from sklearn.cluster import KMeans
    
    # Flatten energy if it's a 2D array
    if energy.ndim > 1:
        energy = energy.ravel()
    
    if by_layers and latents.ndim == 3:
        n_layers = latents.shape[1]
        n_cols = 5
        n_rows = 5
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 25))
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        
        for layer in range(n_layers):
            row = layer // n_cols
            col = layer % n_cols
            latents_layer = latents[:, layer, :]
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(latents_layer)
            
            # PCA for visualization
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(latents_layer)
            
            scatter = axes[row, col].scatter(pca_result[:, 0], pca_result[:, 1], 
                                           c=clusters, cmap='tab10', alpha=0.8, s=50)
            axes[row, col].set_title(f'Layer {layer} Clusters')
            
        # Hide empty subplots
        for i in range(layer + 1, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.suptitle('Layer-wise Cluster Analysis', fontsize=16)
        return fig
    
    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # PCA for visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(latents)
    
    # Plot clusters
    scatter = ax1.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='tab10', alpha=0.8, s=50)
    ax1.set_title('Embedding Clusters')
    ax1.set_xlabel('PCA Component 1')
    ax1.set_ylabel('PCA Component 2')
    legend1 = ax1.legend(*scatter.legend_elements(), title="Clusters")
    ax1.add_artist(legend1)
    
    # Box plot of energy by cluster
    sns.boxplot(x=clusters, y=energy, ax=ax2)
    ax2.set_title('Energy Distribution by Cluster')
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Energy')
    
    plt.suptitle('Catalyst Cluster Analysis', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig

def create_tsne_comparison(latents, energy, highlight_range=False, by_layers=False):
    """
    Create multiple t-SNE visualizations with different perplexity values.
    
    Parameters:
    -----------
    latents : numpy.ndarray
        Embedding vectors of shape (n_samples, embedding_dimension)
    energy : numpy.ndarray
        Energy values of shape (n_samples, 1)
    highlight_range : bool
        If True, highlights points with energy between 1.5 and 2
    """
    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Flatten energy if it's a 2D array
    if energy.ndim > 1:
        energy = energy.ravel()
    
    # Create a mask for samples with energy between 1.5 and 2
    highlight_mask = (energy >= 1.5) & (energy <= 2) if highlight_range else np.ones_like(energy, dtype=bool)
    
    # Different perplexity values
    perplexities = [10, 30, 50, 100]
    
    for idx, perp in enumerate(perplexities):
        row = idx // 2
        col = idx % 2
        
        # Run t-SNE
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
        tsne_result = tsne.fit_transform(latents)
        
        if highlight_range:
            # Plot all points in gray first
            axes[row, col].scatter(tsne_result[:, 0], tsne_result[:, 1], 
                                 c='lightgray', alpha=0.5, s=50, label='Other samples')
            # Plot highlighted points with energy colormap
            im = axes[row, col].scatter(tsne_result[highlight_mask, 0], tsne_result[highlight_mask, 1],
                                      c=energy[highlight_mask], cmap='viridis', alpha=0.8, s=50, 
                                      label='1.5 ≤ Energy ≤ 2')
        else:
            im = axes[row, col].scatter(tsne_result[:, 0], tsne_result[:, 1],
                                      c=energy, cmap='viridis', alpha=0.8, s=50)
        
        axes[row, col].set_title(f't-SNE Visualization (Perplexity={perp})')
        axes[row, col].set_xlabel('t-SNE Component 1')
        axes[row, col].set_ylabel('t-SNE Component 2')
        axes[row, col].legend()
        fig.colorbar(im, ax=axes[row, col], label='Energy')
    
    plt.suptitle('t-SNE Comparison with Different Perplexity Values', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    return fig

# Create output directory if it doesn't exist
os.makedirs('visuals/128', exist_ok=True)

# Example usage with iterations
highlight_options = [True]

# Set filename based on mode
filename = 'logs/2316385/s2ef_predictions.npz' # 153M

# Load and reshape data
data = np.load(filename)
latents = data['latents'].reshape(-1, 21, 128)
energy = data['energy'].reshape(-1, 1)
print(f"Processing mode 128, shape: {latents.shape}, {energy.shape}")

# # Create figure for t-SNE layer visualization
# fig, axes = plt.subplots(7, 3, figsize=(20, 40))  # 7x3 grid for 21 layers
# axes = axes.ravel()  # Flatten axes array for easier indexing

# # Process each layer
# for layer in range(21):
#     latents_layer = latents[:, layer, :]
    
#     # Run t-SNE for each layer
#     tsne = TSNE(n_components=2, perplexity=30, random_state=42)
#     tsne_result = tsne.fit_transform(latents_layer)
    
#     # Plot in the corresponding subplot
#     im = axes[layer].scatter(tsne_result[:, 0], tsne_result[:, 1],
#                            c=energy, cmap='viridis', alpha=0.8, s=50)
    
#     axes[layer].set_title(f'Layer {layer}')
#     axes[layer].set_xlabel('t-SNE Component 1')
#     axes[layer].set_ylabel('t-SNE Component 2')
#     fig.colorbar(im, ax=axes[layer], label='Energy')

#     # Generate other visualizations
#     for highlight in highlight_options:
#         highlight_str = "highlighted" if highlight else "all"
#         print(f"  Creating visualizations for layer {layer} with highlight={highlight}")
        
        # fig1 = visualize_embeddings(latents_layer, energy, method='pca', highlight_range=highlight)
        # fig1.savefig(f'visuals/128/layer{layer}_catalyst_pca_{highlight_str}.png', dpi=300, bbox_inches='tight')
        # plt.close(fig1)
        
        # fig2 = visualize_embeddings(latents_layer, energy, method='tsne', highlight_range=highlight)
        # fig2.savefig(f'visuals/128/layer{layer}_catalyst_tsne_{highlight_str}.png', dpi=300, bbox_inches='tight')
        # plt.close(fig2)
        
    #     fig3 = create_multiple_visualizations(latents_layer, energy, highlight_range=highlight)
    #     fig3.savefig(f'visuals/128/layer{layer}_catalyst_multiple_{highlight_str}.png', dpi=300, bbox_inches='tight')
    #     plt.close(fig3)
    
    #     fig5 = create_tsne_comparison(latents_layer, energy, highlight_range=highlight)
    #     fig5.savefig(f'visuals/128/layer{layer}_catalyst_tsne_comparison_{highlight_str}.png', dpi=300, bbox_inches='tight')
    #     plt.close(fig5)

    # fig4 = analyze_embedding_clusters(latents_layer, energy)
    # fig4.savefig(f'visuals/128/layer{layer}_catalyst_cluster_analysis.png', dpi=300, bbox_inches='tight')
    # plt.close(fig4)

# For layer-wise visualization

# fig1 = visualize_embeddings(latents, energy, method='pca', highlight_range=True, by_layers=True)
# fig1.savefig(f'visuals/128/catalyst_pca_by_layers.png', dpi=300, bbox_inches='tight')
# plt.close(fig1)

fig2 = visualize_embeddings(latents, energy, method='tsne', highlight_range=True, by_layers=True)
fig2.savefig(f'visuals/128/catalyst_tsne_by_layers.png', dpi=300, bbox_inches='tight')
plt.close(fig2)

# fig3 = create_multiple_visualizations(latents, energy, highlight_range=True, by_layers=True)
# fig3.savefig(f'visuals/128/catalyst_multiple_by_layers.png', dpi=300, bbox_inches='tight')
# plt.close(fig3)

# fig4 = create_tsne_comparison(latents, energy, highlight_range=True)
# fig4.savefig(f'visuals/128/catalyst_tsne_comparison_by_layers.png', dpi=300, bbox_inches='tight')
# plt.close(fig4)

# fig5 = analyze_embedding_clusters(latents, energy, by_layers=True)
# fig5.savefig(f'visuals/128/catalyst_cluster_analysis_by_layers.png', dpi=300, bbox_inches='tight')
# plt.close(fig5)