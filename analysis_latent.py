import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

def visualize_embeddings(latents, energy, method='pca', perplexity=50, n_components=2, figsize=(12, 10), highlight_range=False):
    """
    Visualize high-dimensional embeddings with energy values as colors.
    
    Parameters:
    -----------
    latents : numpy.ndarray
        Embedding vectors of shape (n_samples, embedding_dimension)
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
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure containing the plots
    """
    # Flatten energy if it's a 2D array
    if energy.ndim > 1:
        energy = energy.ravel()
    
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

def create_multiple_visualizations(latents, energy, highlight_range=False):
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
    """
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

def analyze_embedding_clusters(latents, energy, n_clusters=5):
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
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(latents)
    
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

def create_tsne_comparison(latents, energy, highlight_range=False):
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


# Example usage with iterations
modes = [128]
highlight_options = [True]

for mode in modes:
    # Set filename based on mode
    if mode == 25:
        filename = 'save_logs/9066/s2ef_predictions.npz' # 31M
    elif mode == 128:
        filename = 'logs/2316385/s2ef_predictions.npz' # 153M
    
    # Load and reshape data
    data = np.load(filename)
    latents = data['latents']
    energy = data['energy'].reshape(-1, 1)
    print(latents.shape, energy.shape)
    print(f"Processing mode {mode}, shape: {latents.shape}, {energy.shape}")
    
    for highlight in highlight_options:
        highlight_str = "highlighted" if highlight else "all"
        print(f"  Creating visualizations with highlight={highlight}")
        
        # Use latents for all visualizations
        #fig1 = visualize_embeddings(latents, energy, method='pca', highlight_range=highlight)
        #fig1.savefig(f'visuals/{mode}/catalyst_pca_{highlight_str}.png', dpi=300, bbox_inches='tight')
        #plt.close(fig1)
        
        fig2 = visualize_embeddings(latents, energy, method='tsne', highlight_range=highlight)
        fig2.savefig(f'visuals/{mode}/catalyst_tsne_{highlight_str}.png', dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        #fig3 = create_multiple_visualizations(latents, energy, highlight_range=highlight)
        #fig3.savefig(f'visuals/{mode}/catalyst_multiple_{highlight_str}.png', dpi=300, bbox_inches='tight')
        #plt.close(fig3)
        
        #fig5 = create_tsne_comparison(latents, energy, highlight_range=highlight)
        #fig5.savefig(f'visuals/{mode}/catalyst_tsne_comparison_{highlight_str}.png', dpi=300, bbox_inches='tight')
        #plt.close(fig5)
    
    # Use latents for cluster analysis too
    #fig4 = analyze_embedding_clusters(latents, energy)
    #fig4.savefig(f'visuals/{mode}/catalyst_cluster_analysis.png', dpi=300, bbox_inches='tight')
    #plt.close(fig4)

