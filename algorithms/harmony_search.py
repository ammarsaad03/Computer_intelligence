import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

class HarmonySearch:
    """Harmony Search Algorithm for dimensionality reduction"""
    def __init__(self, output_dim=2, hm_size=10, hmcr=0.9, par=0.3, bw=0.05,random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        self.output_dim = output_dim
        self.hm_size = hm_size  # Harmony memory size
        self.hmcr = hmcr  # Harmony memory considering rate
        self.par = par    # Pitch adjusting rate
        self.bw = bw      # Bandwidth
        self.harmony_memory = None
        self.memory_scores = None
        self.best_harmony = None
        self.best_score = float('inf')
        self.projected_data = None
        
    def stress(self, X, W):
        """Compute stress (difference between original and projected distances)"""
        Y = X @ W
        orig_dist = np.linalg.norm(X[:, None] - X, axis=2)
        proj_dist = np.linalg.norm(Y[:, None] - Y, axis=2)
        return np.mean((orig_dist - proj_dist) ** 2)
        
    def initialize(self, data):
        input_dim = data.shape[1]
        # Initialize harmony memory with random projection matrices
        self.harmony_memory = [np.random.randn(input_dim, self.output_dim) for _ in range(self.hm_size)]
        self.memory_scores = [self.stress(data, W) for W in self.harmony_memory]
        best_idx = np.argmin(self.memory_scores)
        self.best_harmony = self.harmony_memory[best_idx].copy()
        self.best_score = self.memory_scores[best_idx]
    
    def train(self, data, epochs):
        input_dim = data.shape[1]
        self.initialize(data)
        
        for epoch in range(epochs):
            # Create new harmony (projection matrix)
            new_harmony = np.zeros((input_dim, self.output_dim))
            
            for i in range(input_dim):
                for j in range(self.output_dim):
                    if np.random.rand() < self.hmcr:
                        # Choose from memory
                        idx = np.random.randint(self.hm_size)
                        new_harmony[i,j] = self.harmony_memory[idx][i,j]
                        
                        # Pitch adjustment
                        if np.random.rand() < self.par:
                            new_harmony[i,j] += self.bw * (2 * np.random.rand() - 1)
                    else:
                        # Random creation
                        new_harmony[i,j] = np.random.randn()
            
            # Evaluate new harmony
            new_score = self.stress(data, new_harmony)
            
            # Update harmony memory
            worst_idx = np.argmax(self.memory_scores)
            if new_score < self.memory_scores[worst_idx]:
                self.harmony_memory[worst_idx] = new_harmony.copy()
                self.memory_scores[worst_idx] = new_score
                
                # Update best harmony
                if new_score < self.best_score:
                    self.best_score = new_score
                    self.best_harmony = new_harmony.copy()
        
        # Project the data using the best projection matrix
        self.projected_data = data @ self.best_harmony
    
    def quantization_error(self, data):
        """Calculate mean quantization error similar to other algorithms"""
        if self.best_harmony is None or self.projected_data is None:
            return float('inf')
        
        # Use the projected data to compute distances
        projected_data = self.projected_data
        
        # Compute distances between each point and all other points
        distances = pairwise_distances(projected_data)
        
        # For each point, find its closest neighbor (excluding itself)
        np.fill_diagonal(distances, np.inf)
        min_distances = np.min(distances, axis=1)
        
        # Return the average minimum distance
        return np.mean(min_distances)