import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def create_som_visualization(algo, data, labels, fig, name, q_error, elapsed_time, class_names):
    """Create visualization for Self-Organizing Maps"""
    # SOM visualization
    x_coords = np.arange(algo.grid_size[0])
    y_coords = np.arange(algo.grid_size[1])
    xx, yy = np.meshgrid(x_coords, y_coords)
    
    plt.scatter(xx.flatten(), yy.flatten(), 
               c='gray', alpha=0.3, marker='s', s=100, label='SOM Neurons')
    
    markers = ['o', 's']  # Only need 2 markers for binary classification
    colors = ['r', 'g']
    
    class_points = {label: {'x': [], 'y': []} for label in class_names}
    
    for x, target in zip(data, labels):
        bmu = algo.find_bmu(x)
        class_points[class_names[target]]['x'].append(bmu[0])
        class_points[class_names[target]]['y'].append(bmu[1])
    
    for i, (label, points) in enumerate(class_points.items()):
        plt.scatter(points['x'], points['y'],
                  marker=markers[i % len(markers)],
                  c=colors[i % len(colors)],
                  s=100,
                  label=label)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"{name}\nQE: {q_error:.4f}\nTime: {elapsed_time:.2f}s")
    plt.xlabel("SOM Grid X")
    plt.ylabel("SOM Grid Y")
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.5, algo.grid_size[0] - 0.5)
    plt.ylim(-0.5, algo.grid_size[1] - 0.5)
    plt.tight_layout()

def create_aco_visualization(algo, data, labels, fig, name, q_error, elapsed_time, feature_names, class_names, silhouette=None):
    """Create visualization for Ant Colony Optimization"""
    if algo.projected_data is not None:
        projected_data = algo.projected_data
        
        markers = ['o', 's']
        colors = ['r', 'g']
        
        for i, target_class in enumerate(np.unique(labels)):
            class_indices = np.where(labels == target_class)[0]
            plt.scatter(
                projected_data[class_indices, 0], 
                projected_data[class_indices, 1] if projected_data.shape[1] > 1 else np.zeros_like(projected_data[class_indices, 0]),
                marker=markers[i % len(markers)],
                c=colors[i % len(colors)],
                label=class_names[target_class]
            )
        
        selected_features = [feature_names[i] for i in algo.best_path[:algo.n_features_to_select]]
        plt.legend()
        # Build the title with metrics
        title = f"{name} Dimensionality Reduction\nQE: {q_error:.4f}"
        if silhouette is not None:
            title += f", Silhouette: {silhouette:.4f}"
        title += f"\nTime: {elapsed_time:.2f}s"
        plt.title(title)
        plt.xlabel(f"Feature: {selected_features[0]}")
        plt.ylabel(f"Feature: {selected_features[1]}" if len(selected_features) > 1 else "")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

def create_harmony_visualization(algo, data, labels, fig, name, q_error, elapsed_time, class_names):
    """Create visualization for Harmony Search"""
    if hasattr(algo, 'projected_data') and algo.projected_data is not None:
        projected_data = algo.projected_data
        
        markers = ['o', 's']
        colors = ['r', 'g']
        
        for i, target_class in enumerate(np.unique(labels)):
            class_indices = np.where(labels == target_class)[0]
            plt.scatter(
                projected_data[class_indices, 0], 
                projected_data[class_indices, 1],
                marker=markers[i % len(markers)],
                c=colors[i % len(colors)],
                label=class_names[target_class]
            )
        
        plt.legend()
        plt.title(f"{name} Dimensionality Reduction\nQE: {q_error:.4f}\nTime: {elapsed_time:.2f}s")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

def create_clustering_visualization(algo, data, labels, fig, name, q_error, elapsed_time, class_names):
    """Create visualization for dimensionality reduction algorithms (PSO, GA)"""
    if hasattr(algo, 'projected_data') and algo.projected_data is not None:
        projected_data = algo.projected_data
        
        markers = ['o', 's']
        colors = ['r', 'g']
        
        for i, target_class in enumerate(np.unique(labels)):
            class_indices = np.where(labels == target_class)[0]
            plt.scatter(
                projected_data[class_indices, 0], 
                projected_data[class_indices, 1],
                marker=markers[i % len(markers)],
                c=colors[i % len(colors)],
                label=class_names[target_class]
            )
        
        plt.legend()
        plt.title(f"{name} Dimensionality Reduction\nQE: {q_error:.4f}\nTime: {elapsed_time:.2f}s")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

def create_original_data_visualization(data, labels, class_names, method='pca', perplexity=30, random_seed=None):
    """
    Create visualization of original data using PCA or t-SNE for comparison with algorithm outputs
    
    Args:
        data (np.array): Original high-dimensional data
        labels (np.array): Class labels
        class_names (list): Names of the classes
        method (str): Dimensionality reduction method - 'pca', 'tsne', or 'both'
        perplexity (int): Perplexity parameter for t-SNE (only used if method includes t-SNE)
        
    Returns:
        plt.Figure or tuple: Matplotlib figure(s) with the visualization(s)
    """
    if random_seed is not None:
            np.random.seed(random_seed)
    # Create figure(s) based on method
    if method == 'both':
        # Create two subplots
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        plt.suptitle("Original Data Visualization")
        
        # PCA plot
        pca = PCA(n_components=2, random_state=random_seed)
        data_pca = pca.fit_transform(data)
        
        # t-SNE plot
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_seed)
        data_tsne = tsne.fit_transform(data)
        
        # Plot data with different markers and colors for classes
        markers = ['o', 's']
        colors = ['r', 'g']
        
        for i, target_class in enumerate(np.unique(labels)):
            class_indices = np.where(labels == target_class)[0]
            
            # PCA plot
            axes[0].scatter(
                data_pca[class_indices, 0],
                data_pca[class_indices, 1],
                marker=markers[i % len(markers)],
                c=colors[i % len(colors)],
                label=class_names[target_class]
            )
            
            # t-SNE plot
            axes[1].scatter(
                data_tsne[class_indices, 0],
                data_tsne[class_indices, 1],
                marker=markers[i % len(markers)],
                c=colors[i % len(colors)],
                label=class_names[target_class]
            )
        
        # Add labels and customize
        explained_variance = pca.explained_variance_ratio_
        axes[0].set_title(f"PCA\nExplained Variance: {explained_variance[0]:.2f}, {explained_variance[1]:.2f}")
        axes[0].set_xlabel(f"Principal Component 1 ({explained_variance[0]:.2f})")
        axes[0].set_ylabel(f"Principal Component 2 ({explained_variance[1]:.2f})")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        axes[1].set_title(f"t-SNE\nPerplexity: {perplexity}")
        axes[1].set_xlabel("t-SNE Component 1")
        axes[1].set_ylabel("t-SNE Component 2")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        
    elif method == 'tsne':
        # Create figure for t-SNE
        fig = plt.figure(figsize=(10, 8))
        
        # Apply t-SNE for visualization
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_seed)
        data_tsne = tsne.fit_transform(data)
        
        # Plot with different markers and colors for classes
        markers = ['o', 's']
        colors = ['r', 'g']
        
        for i, target_class in enumerate(np.unique(labels)):
            class_indices = np.where(labels == target_class)[0]
            plt.scatter(
                data_tsne[class_indices, 0],
                data_tsne[class_indices, 1],
                marker=markers[i % len(markers)],
                c=colors[i % len(colors)],
                label=class_names[target_class]
            )
        
        # Add labels and title
        plt.title(f"Original Data (t-SNE)\nPerplexity: {perplexity}")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
    else:  # Default to PCA
        # Create figure for PCA
        fig = plt.figure(figsize=(10, 8))
        
        # Apply PCA for visualization
        pca = PCA(n_components=2, random_state=random_seed)
        data_pca = pca.fit_transform(data)
        
        # Plot with different markers and colors for classes
        markers = ['o', 's']
        colors = ['r', 'g']
        
        for i, target_class in enumerate(np.unique(labels)):
            class_indices = np.where(labels == target_class)[0]
            plt.scatter(
                data_pca[class_indices, 0],
                data_pca[class_indices, 1],
                marker=markers[i % len(markers)],
                c=colors[i % len(colors)],
                label=class_names[target_class]
            )
        
        # Add labels and title
        explained_variance = pca.explained_variance_ratio_
        plt.title(f"Original Data (PCA)\nExplained Variance: {explained_variance[0]:.2f}, {explained_variance[1]:.2f}")
        plt.xlabel(f"Principal Component 1 ({explained_variance[0]:.2f})")
        plt.ylabel(f"Principal Component 2 ({explained_variance[1]:.2f})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
    return fig