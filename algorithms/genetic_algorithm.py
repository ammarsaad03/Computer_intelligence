import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

class GeneticAlgorithm:
    """Genetic Algorithm for dimensionality reduction"""
    def __init__(self, output_dim=2, pop_size=20, crossover_rate=0.8, mutation_rate=0.1, elite_size=2):
        self.output_dim = output_dim
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.population = None
        self.fitness_scores = None
        self.best_solution = None
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
        # Initialize population with random projection matrices
        self.population = []
        for _ in range(self.pop_size):
            # Random projection matrix
            W = np.random.randn(input_dim, self.output_dim)
            self.population.append(W)
        
        # Calculate fitness for each individual
        self.fitness_scores = [self.stress(data, ind) for ind in self.population]
        
        # Track best solution
        best_idx = np.argmin(self.fitness_scores)
        self.best_solution = self.population[best_idx].copy()
        self.best_score = self.fitness_scores[best_idx]
    
    def selection(self):
        """Tournament selection"""
        selected = []
        for _ in range(self.pop_size - self.elite_size):
            # Select 3 random individuals and pick the best one
            indices = np.random.choice(len(self.population), 3, replace=False)
            tournament = [(i, self.fitness_scores[i]) for i in indices]
            winner_idx = min(tournament, key=lambda x: x[1])[0]
            selected.append(self.population[winner_idx].copy())
        return selected
    
    def crossover(self, parent1, parent2):
        """Uniform crossover for matrices"""
        child = parent1.copy()
        
        # Create crossover mask (True/False)
        mask = np.random.rand(*parent1.shape) < 0.5
        
        # Apply mask for crossover
        child[mask] = parent2[mask]
        
        return child
    
    def mutate(self, individual):
        """Gaussian mutation for matrices"""
        mutation_mask = np.random.rand(*individual.shape) < self.mutation_rate
        
        # Only mutate selected elements
        if mutation_mask.any():
            # Gaussian noise with adaptive scale based on the value range
            scale = 0.1 * np.std(individual)
            individual[mutation_mask] += np.random.normal(0, scale, size=np.sum(mutation_mask))
        
        return individual
    
    def train(self, data, epochs):
        input_dim = data.shape[1]
        self.initialize(data)
        
        for epoch in range(epochs):
            # 1. Elitism: Keep the best individuals
            elite_indices = np.argsort(self.fitness_scores)[:self.elite_size]
            elite = [self.population[i].copy() for i in elite_indices]
            
            # 2. Selection
            selected = self.selection()
            
            # 3. Crossover
            offspring = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    if np.random.rand() < self.crossover_rate:
                        offspring.append(self.crossover(selected[i], selected[i+1]))
                        offspring.append(self.crossover(selected[i+1], selected[i]))
                    else:
                        offspring.append(selected[i].copy())
                        offspring.append(selected[i+1].copy())
                else:
                    offspring.append(selected[i].copy())
            
            # 4. Mutation
            for i in range(len(offspring)):
                offspring[i] = self.mutate(offspring[i])
            
            # 5. Create new population
            self.population = elite + offspring
            self.fitness_scores = [self.stress(data, ind) for ind in self.population]
            
            # 6. Update best solution
            current_best_idx = np.argmin(self.fitness_scores)
            if self.fitness_scores[current_best_idx] < self.best_score:
                self.best_score = self.fitness_scores[current_best_idx]
                self.best_solution = self.population[current_best_idx].copy()
        
        # Project the data using the best projection matrix
        self.projected_data = data @ self.best_solution
    
    def quantization_error(self, data):
        """Calculate mean quantization error similar to other algorithms"""
        if self.best_solution is None or self.projected_data is None:
            return float('inf')
        
        # Compute distances between each point and all other points in projected space
        distances = pairwise_distances(self.projected_data)
        
        # For each point, find its closest neighbor (excluding itself)
        np.fill_diagonal(distances, np.inf)
        min_distances = np.min(distances, axis=1)
        
        # Return the average minimum distance
        return np.mean(min_distances)