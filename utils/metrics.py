import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

def trustworthiness(original_data, embedded_data, n_neighbors=5):
    """
    Calculate the trustworthiness of a dimensionality reduction.    
    Args:
        original_data (np.array): Original high-dimensional data
        embedded_data (np.array): Reduced low-dimensional data
        n_neighbors (int): Number of neighbors to consider
        
    Returns:
        float: Trustworthiness score (between 0 and 1)
    """
    # Get nearest neighbors in the original space
    nn_orig = NearestNeighbors(n_neighbors=n_neighbors+1).fit(original_data)
    dist_orig, ind_orig = nn_orig.kneighbors()
    
    # Remove the point itself from neighbors
    ind_orig = ind_orig[:, 1:]
    
    # Get nearest neighbors in the embedded space
    nn_embed = NearestNeighbors(n_neighbors=n_neighbors+1).fit(embedded_data)
    dist_embed, ind_embed = nn_embed.kneighbors()
    
    # Remove the point itself from neighbors
    ind_embed = ind_embed[:, 1:]
    
    # Calculate trustworthiness
    n_samples = original_data.shape[0]
    trustworthiness_sum = 0
    
    for i in range(n_samples):
        # For each point, find which points in its embedded neighborhood
        # were not in its original neighborhood
        embed_neighbors = set(ind_embed[i])
        orig_neighbors = set(ind_orig[i])
        
        # Find neighbors in embedded space that weren't neighbors in original space
        false_neighbors = embed_neighbors - orig_neighbors
        
        # Calculate the rank of these false neighbors in the original space
        for j in false_neighbors:
            # Find rank of j in original space distances from point i
            # (rank = position in sorted list of distances)
            orig_distances = np.linalg.norm(original_data - original_data[i], axis=1)
            sorted_indices = np.argsort(orig_distances)
            rank = np.where(sorted_indices == j)[0][0]
            
            trustworthiness_sum += (rank - n_neighbors)
    
    # Normalize the result
    normalization = 2 / (n_samples * n_neighbors * (2 * n_samples - 3 * n_neighbors - 1))
    trustworthiness_score = 1 - normalization * trustworthiness_sum
    
    return max(0, min(1, trustworthiness_score))  # Clamp to [0, 1]


def calculate_silhouette(data, labels):
    """
    Calculate silhouette score for clustering results.
    
    The silhouette score measures how well-separated the clusters are.
    Values range from -1 to 1, with 1 being the best.
    
    Args:
        data (np.array): Data points used for silhouette calculation
                         (can be original or reduced dimensions)
        labels (np.array): Cluster labels for each data point
    
    Returns:
        float: Silhouette score between -1 and 1
    """
    # We need at least 2 clusters and each cluster should have more than 1 sample
    n_clusters = len(np.unique(labels))
    if n_clusters < 2:
        return 0 
    
    # Check if we have enough samples in each cluster
    for label in np.unique(labels):
        if np.sum(labels == label) <= 1:
            return 0  # Can't calculate silhouette if a cluster has only one sample
    
    try:
        # Calculate silhouette score
        return silhouette_score(data, labels)
    except Exception as e:
        print(f"Error calculating silhouette score: {e}")
        return 0