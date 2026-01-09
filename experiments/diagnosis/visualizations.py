"""
Visualization Utilities for ORCA-DeSTA Diagnosis

Implements visualization tools for the paper:
- t-SNE visualization (Observation 3: Content-Style Entanglement)
- PCA variance curves
- Group similarity heatmaps
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Dict, List, Optional, Union
import os


def plot_tsne_by_attribute(
    features: np.ndarray,
    labels: np.ndarray,
    attribute_name: str,
    save_path: Optional[str] = None,
    perplexity: int = 30,
    n_iter: int = 1000,
    figsize: tuple = (10, 8),
    title: Optional[str] = None,
    colormap: str = "tab10"
) -> plt.Figure:
    """
    Create t-SNE visualization colored by attribute.
    
    Args:
        features: [N, D] feature matrix
        labels: [N] attribute labels (categorical)
        attribute_name: Name of the attribute for title/legend
        save_path: Path to save figure (optional)
        perplexity: t-SNE perplexity parameter
        n_iter: Number of t-SNE iterations
        figsize: Figure size
        title: Custom title (default: auto-generated)
        colormap: Matplotlib colormap name
        
    Returns:
        Matplotlib figure
    """
    # Flatten features if needed
    if len(features.shape) > 2:
        features = features.reshape(features.shape[0], -1)
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    embedded = tsne.fit_transform(features)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique labels and colors
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap(colormap)
    colors = [cmap(i / len(unique_labels)) for i in range(len(unique_labels))]
    
    # Plot each class
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            embedded[mask, 0], embedded[mask, 1],
            c=[colors[i]], label=str(label),
            alpha=0.6, s=20
        )
    
    if title is None:
        title = f"t-SNE Visualization by {attribute_name}"
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(title=attribute_name, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved t-SNE plot to {save_path}")
    
    return fig


def plot_tsne_comparison(
    features_dict: Dict[str, np.ndarray],
    labels: np.ndarray,
    attribute_name: str,
    save_path: Optional[str] = None,
    perplexity: int = 30,
    figsize_per_model: tuple = (6, 5)
) -> plt.Figure:
    """
    Create side-by-side t-SNE comparison of multiple models.
    
    Args:
        features_dict: Dict mapping model name to features
        labels: [N] attribute labels (same for all models)
        attribute_name: Name of the attribute
        save_path: Path to save figure
        perplexity: t-SNE perplexity
        figsize_per_model: Size per subplot
        
    Returns:
        Matplotlib figure
    """
    n_models = len(features_dict)
    fig, axes = plt.subplots(1, n_models, 
                             figsize=(figsize_per_model[0] * n_models, figsize_per_model[1]))
    
    if n_models == 1:
        axes = [axes]
    
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab10")
    colors = {label: cmap(i / len(unique_labels)) for i, label in enumerate(unique_labels)}
    
    for ax, (model_name, features) in zip(axes, features_dict.items()):
        # Flatten features
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)
        
        # t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        embedded = tsne.fit_transform(features)
        
        # Plot
        for label in unique_labels:
            mask = labels == label
            ax.scatter(embedded[mask, 0], embedded[mask, 1],
                      c=[colors[label]], label=str(label), alpha=0.6, s=15)
        
        ax.set_title(model_name, fontsize=12)
        ax.set_xlabel("t-SNE Dim 1")
        ax.set_ylabel("t-SNE Dim 2")
    
    # Single legend
    handles, labels_legend = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels_legend, title=attribute_name,
               loc='center right', bbox_to_anchor=(1.12, 0.5))
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_pca_variance_curve(
    variance_dict: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    max_components: int = 20,
    figsize: tuple = (10, 5)
) -> plt.Figure:
    """
    Plot cumulative PCA variance curves for model comparison.
    
    Args:
        variance_dict: Dict mapping model name to cumulative variance array
        save_path: Path to save figure
        max_components: Maximum number of components to plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(variance_dict)))
    
    for (name, variance), color in zip(variance_dict.items(), colors):
        n_comp = min(len(variance), max_components)
        x = np.arange(1, n_comp + 1)
        
        # Cumulative variance
        ax1.plot(x, variance[:n_comp] * 100, 'o-', label=name, color=color, markersize=4)
        
        # Individual variance (first few)
        if len(variance) > 1:
            individual = np.diff(np.concatenate([[0], variance]))
            ax2.bar(x[:10] + (list(variance_dict.keys()).index(name) - len(variance_dict)/2) * 0.15,
                   individual[:10] * 100, width=0.15, label=name, color=color, alpha=0.8)
    
    ax1.set_xlabel("Number of Principal Components")
    ax1.set_ylabel("Cumulative Explained Variance (%)")
    ax1.set_title("PCA Explained Variance (Cumulative)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='90% threshold')
    
    ax2.set_xlabel("Principal Component")
    ax2.set_ylabel("Explained Variance (%)")
    ax2.set_title("Individual Component Variance (Top 10)")
    ax2.legend()
    ax2.set_xticks(range(1, 11))
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_group_similarity_heatmap(
    similarity_matrix: np.ndarray,
    group_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6),
    title: str = "Inter-Group Cosine Similarity"
) -> plt.Figure:
    """
    Create heatmap of group centroid similarities.
    
    Args:
        similarity_matrix: [num_groups, num_groups] similarity matrix
        group_names: Names for each group (optional)
        save_path: Path to save figure
        figsize: Figure size
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    n_groups = similarity_matrix.shape[0]
    
    if group_names is None:
        group_names = [f"G{i+1}" for i in range(n_groups)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(similarity_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Cosine Similarity")
    
    # Labels
    ax.set_xticks(range(n_groups))
    ax.set_yticks(range(n_groups))
    ax.set_xticklabels(group_names)
    ax.set_yticklabels(group_names)
    
    # Add value annotations
    for i in range(n_groups):
        for j in range(n_groups):
            value = similarity_matrix[i, j]
            color = 'white' if abs(value) > 0.5 else 'black'
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", 
                   color=color, fontsize=8)
    
    ax.set_title(title)
    ax.set_xlabel("Group")
    ax.set_ylabel("Group")
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_metrics_comparison_table(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 4)
) -> plt.Figure:
    """
    Create a table visualization of key metrics across models.
    
    Args:
        metrics_dict: Dict mapping model name to metrics dict
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Extract data
    models = list(metrics_dict.keys())
    metrics = list(metrics_dict[models[0]].keys())
    
    # Create table data
    cell_text = []
    for model in models:
        row = [f"{metrics_dict[model][m]:.3f}" if isinstance(metrics_dict[model][m], float)
               else str(metrics_dict[model][m]) for m in metrics]
        cell_text.append(row)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    
    table = ax.table(
        cellText=cell_text,
        rowLabels=models,
        colLabels=metrics,
        loc='center',
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color header
    for j, key in enumerate(metrics):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', weight='bold')
    
    plt.title("Model Comparison Metrics", fontsize=14, pad=20)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    # Test visualizations with synthetic data
    print("Visualization Module Test")
    
    np.random.seed(42)
    n_samples = 500
    
    # Create synthetic clustered data
    from sklearn.datasets import make_blobs
    
    features, labels = make_blobs(n_samples=n_samples, n_features=64, centers=4, random_state=42)
    
    # Test t-SNE
    fig = plot_tsne_by_attribute(features, labels, "Emotion", save_path=None)
    plt.show()
    
    # Test group similarity heatmap
    sim_matrix = np.random.rand(8, 8) * 0.2 - 0.1
    np.fill_diagonal(sim_matrix, 1.0)
    sim_matrix = (sim_matrix + sim_matrix.T) / 2  # Make symmetric
    
    fig = plot_group_similarity_heatmap(sim_matrix)
    plt.show()
    
    print("Visualization tests complete!")
