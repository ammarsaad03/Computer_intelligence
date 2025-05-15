import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from utils.metrics import calculate_silhouette

def train_algorithm(name, algorithm, data, labels, epochs, feature_names, class_names,random_seed=None, progress_callback=None):
    # Set seeds at the beginning of each training run
    if random_seed is not None:
        np.random.seed(random_seed)
    """
    Standardized wrapper for training nature-inspired algorithms
    
    Args:
        name (str): Name of the algorithm
        algorithm (object): Algorithm instance
        data (np.array): Input data
        labels (np.array): Class labels
        epochs (int): Number of training iterations
        feature_names (list): Names of the features
        class_names (list): Names of the classes
        progress_callback (function): Optional callback for progress updates
    
    Returns:
        dict: Dictionary with results including trained algorithm, error, time, and plot
    """
    # Start timing
    start_time = time.time()
    
    # Progress update
    if progress_callback:
        progress_callback(f"Initializing {name}...")
    
    # Training step
    if name == "Ant Colony Optimization":
        # ACO needs labels for better feature selection
        algorithm.train(data, epochs, labels)
    else:
        # Standard training for other algorithms
        algorithm.train(data, epochs)
    
    # Calculate quantization error
    q_error = algorithm.quantization_error(data)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    silhouette_value = calculate_silhouette(algorithm.projected_data, labels)
    # Create visualization
    fig = plt.figure(figsize=(10, 8))
    
    from utils.visualization import (
        create_som_visualization,
        create_aco_visualization,
        create_harmony_visualization,
        create_clustering_visualization
    )
    if name == "Standard SOM":
        create_som_visualization(algorithm, data, labels, fig, name, q_error, elapsed_time, class_names)
    elif name == "Ant Colony Optimization":
        create_aco_visualization(algorithm, data, labels, fig, name, q_error, elapsed_time, feature_names, class_names,silhouette_value)
    elif name == "Harmony Search":
        create_harmony_visualization(algorithm, data, labels, fig, name, q_error, elapsed_time, class_names)
    else:
        create_clustering_visualization(algorithm, data, labels, fig, name, q_error, elapsed_time, class_names)
    
    # Return results
    return {
        'algorithm': algorithm,
        'quantization_error': q_error,
        'silhouette_score': silhouette_value,
        'time': elapsed_time,
        'plot': fig,
        'name': name
    }
