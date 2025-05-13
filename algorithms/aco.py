import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

class ACO:
    """Ant Colony Optimization for feature selection and dimensionality reduction"""
    def __init__(self, n_ants=10, alpha=1, beta=2, evaporation_rate=0.5, q=100, n_features_to_select=2):
        self.n_ants = n_ants
        self.alpha = alpha  # Pheromone importance
        self.beta = beta    # Heuristic information importance
        self.evaporation_rate = evaporation_rate
        self.q = q  # Pheromone deposit amount
        self.n_features_to_select = n_features_to_select
        self.pheromone_matrix = None
        self.best_path = None
        self.best_fitness = float('inf')
        self.reduced_data = None
        
    def initialize(self, data):
        self.n_features = data.shape[1]
        self.pheromone_matrix = np.ones((self.n_features, self.n_features))
        self.dist_matrix = pairwise_distances(data.T)  # Distance between features
        
    def select_next_node(self, path, graph):
        current_node = path[-1]
        pheromones = self.pheromone_matrix[current_node]
        probabilities = []

        for i in range(self.n_features):
            if i not in path:
                pheromone = pheromones[i] ** self.alpha
                distance = self.dist_matrix[current_node, i] ** self.beta
                probabilities.append((pheromone / distance) if distance > 0 else 0)
            else:
                probabilities.append(0)

        probabilities = np.array(probabilities)
        if probabilities.sum() > 0:
            probabilities = probabilities / probabilities.sum()
            return np.random.choice(range(self.n_features), p=probabilities)
        else:
            # If all probabilities are zero, select randomly from unvisited nodes
            unvisited = [i for i in range(self.n_features) if i not in path]
            if unvisited:
                return np.random.choice(unvisited)
            return np.random.randint(0, self.n_features)
    
    def calculate_fitness(self, path, data, labels=None):
        if len(path) < self.n_features_to_select:
            return float('inf')
            
        selected_features = data[:, path[:self.n_features_to_select]]
        
        if labels is not None and len(np.unique(labels)) > 1:
            # If labels are provided, use classification accuracy
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            try:
                scores = cross_val_score(clf, selected_features, labels, cv=min(3, len(np.unique(labels))))
                return 1 - scores.mean()  # Lower is better
            except:
                # Fallback if classification fails
                return self.calculate_internal_fitness(selected_features)
        else:
            # Otherwise use internal evaluation
            return self.calculate_internal_fitness(selected_features)
            
    def calculate_internal_fitness(self, selected_features):
        # Silhouette score requires at least 2 samples and 2 clusters
        if selected_features.shape[0] < 2 or selected_features.shape[1] < 1:
            return float('inf')
            
        # Evaluating feature quality without labels using variance
        # Higher variance in features is generally better for dimensionality reduction
        return 1 / (np.var(selected_features) + 1e-10)  # Lower is better
    
    def train(self, data, epochs, labels=None):
        self.initialize(data)
        
        for epoch in range(epochs):
            ants_paths = []
            ants_fitness = []
            
            for ant in range(self.n_ants):
                # Start from a random feature
                path = [np.random.randint(0, self.n_features)]
                
                # Build path
                while len(path) < min(self.n_features, 10):  # Limit path length
                    next_node = self.select_next_node(path, data)
                    path.append(next_node)
                
                fitness = self.calculate_fitness(path, data, labels)
                ants_paths.append(path)
                ants_fitness.append(fitness)
                
                # Update best solution
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_path = path.copy()
            
            # Evaporate pheromones
            self.pheromone_matrix *= (1 - self.evaporation_rate)
            
            # Update pheromones based on ant paths
            for path, fitness in zip(ants_paths, ants_fitness):
                if fitness < float('inf'):
                    for i in range(len(path) - 1):
                        start, end = path[i], path[i + 1]
                        self.pheromone_matrix[start, end] += self.q / (fitness + 1e-10)
        
        # Project data to best features
        if self.best_path and len(self.best_path) >= self.n_features_to_select:
            self.reduced_data = data[:, self.best_path[:self.n_features_to_select]]
    
    def quantization_error(self, data):
        """Calculate mean quantization error similar to other algorithms"""
        if self.best_path is None or self.reduced_data is None:
            return float('inf')
        
        # If we have selected features, use them to compute distances
        selected_features = data[:, self.best_path[:self.n_features_to_select]]
        
        # Compute distances between each point and all other points
        distances = pairwise_distances(selected_features)
        
        # For each point, find its closest neighbor (excluding itself)
        np.fill_diagonal(distances, np.inf)
        min_distances = np.min(distances, axis=1)
        
        # Return the average minimum distance
        return np.mean(min_distances)