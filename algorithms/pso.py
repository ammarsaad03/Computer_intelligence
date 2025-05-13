import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

class PSO:
    """Particle Swarm Optimization for dimensionality reduction"""
    def __init__(self, output_dim=2, n_particles=10, w=0.7, c1=1.5, c2=1.5):
        self.output_dim = output_dim
        self.n_particles = n_particles
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive coefficient
        self.c2 = c2  # Social coefficient
        self.particles = None
        self.velocities = None
        self.pbest = None
        self.pbest_scores = None
        self.gbest = None
        self.gbest_score = float('inf')
        self.projected_data = None
        
    def stress(self, X, W):
        """Compute stress (difference between original and projected distances)"""
        Y = X @ W
        orig_dist = np.linalg.norm(X[:, None] - X, axis=2)
        proj_dist = np.linalg.norm(Y[:, None] - Y, axis=2)
        return np.mean((orig_dist - proj_dist) ** 2)
        
    def initialize(self, data):
        input_dim = data.shape[1]
        # Each particle represents a projection matrix
        self.particles = [np.random.randn(input_dim, self.output_dim) for _ in range(self.n_particles)]
        self.velocities = [np.zeros((input_dim, self.output_dim)) for _ in range(self.n_particles)]
        
        # Initialize personal best
        self.pbest = [p.copy() for p in self.particles]
        self.pbest_scores = [self.stress(data, p) for p in self.particles]
        
        # Initialize global best
        best_idx = np.argmin(self.pbest_scores)
        self.gbest = self.pbest[best_idx].copy()
        self.gbest_score = self.pbest_scores[best_idx]
    
    def train(self, data, epochs):
        input_dim = data.shape[1]
        self.initialize(data)
        
        for epoch in range(epochs):
            for i in range(self.n_particles):
                # Calculate fitness
                current_score = self.stress(data, self.particles[i])
                
                # Update personal best
                if current_score < self.pbest_scores[i]:
                    self.pbest[i] = self.particles[i].copy()
                    self.pbest_scores[i] = current_score
                
                # Update global best
                if current_score < self.gbest_score:
                    self.gbest = self.particles[i].copy()
                    self.gbest_score = current_score
            
            # Update velocities and positions
            for i in range(self.n_particles):
                # Generate random coefficients
                r1 = np.random.rand(input_dim, self.output_dim)
                r2 = np.random.rand(input_dim, self.output_dim)
                
                # Update velocity
                cognitive = self.c1 * r1 * (self.pbest[i] - self.particles[i])
                social = self.c2 * r2 * (self.gbest - self.particles[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive + social
                
                # Update position
                self.particles[i] += self.velocities[i]
        
        # Project the data using the best projection matrix
        self.projected_data = data @ self.gbest
    
    def quantization_error(self, data):
        """Calculate mean quantization error similar to other algorithms"""
        if self.gbest is None or self.projected_data is None:
            return float('inf')
        
        # Compute distances between each point and all other points in projected space
        distances = pairwise_distances(self.projected_data)
        
        # For each point, find its closest neighbor (excluding itself)
        np.fill_diagonal(distances, np.inf)
        min_distances = np.min(distances, axis=1)
        
        # Return the average minimum distance
        return np.mean(min_distances)