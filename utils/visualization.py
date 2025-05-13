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

def create_aco_visualization(algo, data, labels, fig, name, q_error, elapsed_time, feature_names, class_names):
    """Create visualization for Ant Colony Optimization"""
    if hasattr(algo, 'reduced_data') and algo.reduced_data is not None:
        reduced_data = algo.reduced_data
        
        markers = ['o', 's']
        colors = ['r', 'g']
        
        for i, target_class in enumerate(np.unique(labels)):
            class_indices = np.where(labels == target_class)[0]
            plt.scatter(
                reduced_data[class_indices, 0], 
                reduced_data[class_indices, 1] if reduced_data.shape[1] > 1 else np.zeros_like(reduced_data[class_indices, 0]),
                marker=markers[i % len(markers)],
                c=colors[i % len(colors)],
                label=class_names[target_class]
            )
        
        selected_features = [feature_names[i] for i in algo.best_path[:algo.n_features_to_select]]
        plt.legend()
        plt.title(f"{name} Feature Selection\nFeatures: {selected_features}\nQE: {q_error:.4f}\nTime: {elapsed_time:.2f}s")
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