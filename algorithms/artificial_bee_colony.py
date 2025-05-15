import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

class ArtificialBeeColony:
    def __init__(self, output_dim=2, n_bees=20, limit=20, max_cycles=100,random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
          
        self.output_dim = output_dim
        self.n_bees = n_bees  # Number of employed bees (equal to food sources)
        self.limit = limit    # Limit of trials before abandoning a food source
        self.max_cycles = max_cycles
        self.food_sources = None
        self.food_fitness = None
        self.trials = None
        self.best_source = None
        self.best_fitness = float('inf')
        self.projected_data = None
        
    def stress(self, X, W):
        """Compute stress (difference between original and projected distances)"""
        Y = X @ W
        orig_dist = np.linalg.norm(X[:, None] - X, axis=2)
        proj_dist = np.linalg.norm(Y[:, None] - Y, axis=2)
        return np.mean((orig_dist - proj_dist) ** 2)
        
    def initialize(self, data):
        input_dim = data.shape[1]
        # Initialize food sources randomly
        self.food_sources = [np.random.randn(input_dim, self.output_dim) for _ in range(self.n_bees)]
        self.food_fitness = [self.stress(data, source) for source in self.food_sources]
        self.trials = np.zeros(self.n_bees)
        
        # Find best source
        best_idx = np.argmin(self.food_fitness)
        self.best_source = self.food_sources[best_idx].copy()
        self.best_fitness = self.food_fitness[best_idx]
    
    def employed_bee_phase(self, data):
        """Employed bees search for new food sources near current ones"""
        for i in range(self.n_bees):
            # Generate a neighbor solution by modifying one random dimension
            new_source = self.food_sources[i].copy()
            
            # Choose a random dimension to modify
            a, b = np.random.randint(0, new_source.shape[0]), np.random.randint(0, new_source.shape[1])
            
            # Choose a random neighbor for the dimension modification (excluding self)
            k = i
            while k == i:
                k = np.random.randint(0, self.n_bees)
                
            # Apply modification
            phi = np.random.uniform(-1, 1)
            new_source[a, b] = self.food_sources[i][a, b] + phi * (self.food_sources[i][a, b] - self.food_sources[k][a, b])
            
            # Evaluate new source
            new_fitness = self.stress(data, new_source)
            
            # Greedy selection
            if new_fitness < self.food_fitness[i]:
                self.food_sources[i] = new_source
                self.food_fitness[i] = new_fitness
                self.trials[i] = 0
                
                # Update best solution if needed
                if new_fitness < self.best_fitness:
                    self.best_source = new_source.copy()
                    self.best_fitness = new_fitness
            else:
                self.trials[i] += 1
    
    def onlooker_bee_phase(self, data):
        """Onlooker bees select food sources based on their quality"""
        # Calculate selection probabilities (lower fitness = higher probability)
        max_fitness = max(self.food_fitness)
        fitness_values = [max_fitness - fitness for fitness in self.food_fitness]
        total_fitness = sum(fitness_values) or 1  # Avoid division by zero
        probabilities = [fitness / total_fitness for fitness in fitness_values]
        
        # Onlooker bees select sources and improve them
        i = 0
        t = 0
        while t < self.n_bees:
            if np.random.random() < probabilities[i]:
                t += 1
                
                # Generate a neighbor solution
                new_source = self.food_sources[i].copy()
                
                # Choose a random dimension to modify
                a, b = np.random.randint(0, new_source.shape[0]), np.random.randint(0, new_source.shape[1])
                
                # Choose a random neighbor
                k = i
                while k == i:
                    k = np.random.randint(0, self.n_bees)
                    
                # Apply modification
                phi = np.random.uniform(-1, 1)
                new_source[a, b] = self.food_sources[i][a, b] + phi * (self.food_sources[i][a, b] - self.food_sources[k][a, b])
                
                # Evaluate new source
                new_fitness = self.stress(data, new_source)
                
                # Greedy selection
                if new_fitness < self.food_fitness[i]:
                    self.food_sources[i] = new_source
                    self.food_fitness[i] = new_fitness
                    self.trials[i] = 0
                    
                    # Update best solution if needed
                    if new_fitness < self.best_fitness:
                        self.best_source = new_source.copy()
                        self.best_fitness = new_fitness
                else:
                    self.trials[i] += 1
            
            i = (i + 1) % self.n_bees
    
    def scout_bee_phase(self, data):
        """Scout bees replace abandoned sources with random ones"""
        # Find abandoned sources
        for i in range(self.n_bees):
            if self.trials[i] > self.limit:
                # Generate new random source
                input_dim = data.shape[1]
                self.food_sources[i] = np.random.randn(input_dim, self.output_dim)
                self.food_fitness[i] = self.stress(data, self.food_sources[i])
                self.trials[i] = 0
                
                # Update best solution if needed
                if self.food_fitness[i] < self.best_fitness:
                    self.best_source = self.food_sources[i].copy()
                    self.best_fitness = self.food_fitness[i]
    
    def train(self, data, epochs):
        """Train the ABC algorithm for dimensionality reduction"""
        self.initialize(data)
        best_fitness_history = []
        patience = 10
        for cycle in range(min(epochs, self.max_cycles)):
            # Employed bee phase
            self.employed_bee_phase(data)
            
            # Onlooker bee phase
            self.onlooker_bee_phase(data)
            
            # Scout bee phase
            self.scout_bee_phase(data)
            best_fitness_history.append(self.best_fitness)
            # Early stopping
            if len(best_fitness_history) > patience:
                if all(abs(self.best_fitness - f) < 1e-4 for f in best_fitness_history[-patience:]):
                    break
        
        # Project the data using the best projection matrix
        self.projected_data = data @ self.best_source
    
    def quantization_error(self, data):
        """Calculate mean quantization error similar to other algorithms"""
        if self.best_source is None or self.projected_data is None:
            return float('inf')
        
        # Compute distances between each point and all other points in projected space
        distances = pairwise_distances(self.projected_data)
        
        # For each point, find its closest neighbor (excluding itself)
        np.fill_diagonal(distances, np.inf)
        min_distances = np.min(distances, axis=1)
        
        # Return the average minimum distance
        return np.mean(min_distances)