import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

class StandardSOM:
    """
    Standard Self-Organizing Map (SOM) implementation
    
    Args:
        grid_size (int or tuple): If int, creates square grid of that size. If tuple, dimensions of the 2D neuron grid (rows, cols)
        input_dim (int): Dimension of input data
        sigma (float): Initial neighborhood radius
        lr (float): Initial learning rate
    """
    def __init__(self, grid_size, input_dim, sigma=1.0, lr=0.5):
        # Handle both integer and tuple grid_size
        if isinstance(grid_size, int):
            grid_size = (grid_size, grid_size)
        self.grid = np.random.rand(grid_size[0], grid_size[1], input_dim)
        self.sigma = sigma  # Neighborhood radius
        self.lr = lr        # Learning rate
        self.grid_size = grid_size
        self.input_dim = input_dim
        
    def find_bmu(self, x):
        """Find Best Matching Unit (BMU) for input vector x"""
        distances = np.linalg.norm(self.grid - x, axis=2)
        return np.unravel_index(np.argmin(distances), distances.shape)
    
    def neighborhood(self, distance, sigma):
        """Gaussian neighborhood function"""
        return np.exp(-distance**2 / (2 * sigma**2))
    
    def train(self, data, epochs):
        """
        Train the SOM
        
        Args:
            data (np.array): Input data of shape (n_samples, input_dim)
            epochs (int): Number of training iterations
        """
        for epoch in range(epochs):
            # Decay parameters
            current_sigma = self.sigma * (1 - epoch/epochs)
            current_lr = self.lr * (1 - epoch/epochs)
            
            for x in data:
                # 1. Find BMU
                bmu = self.find_bmu(x)
                
                # 2. Update weights
                for i in range(self.grid_size[0]):
                    for j in range(self.grid_size[1]):
                        # Distance from neuron (i,j) to BMU
                        distance_to_bmu = np.linalg.norm(np.array([i,j]) - np.array(bmu))
                        influence = self.neighborhood(distance_to_bmu, current_sigma)
                        
                        # Weight update rule
                        self.grid[i,j] += current_lr * influence * (x - self.grid[i,j])
    
    def quantization_error(self, data):
        """Calculate mean quantization error consistent with ACO and Harmony Search"""
        # Map data points to their BMU weight vectors
        bmu_weights = np.array([self.grid[self.find_bmu(x)] for x in data])
        
        # Compute pairwise distances between the mapped representations
        distances = pairwise_distances(bmu_weights)
        
        # For each point, find its closest neighbor (excluding itself)
        np.fill_diagonal(distances, np.inf)
        min_distances = np.min(distances, axis=1)
        
        # Return the average minimum distance
        return np.mean(min_distances)