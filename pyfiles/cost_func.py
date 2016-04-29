import numpy as np
def cost(dist):
    """ Calculate the cost of data with respect to the current centroids
    Parameters:
       dist                     distance matrix between data and current centroids
    
    Returns:
       cost                     the normalized constant in the distribution 
    """
    min_dist = np.zeros(dist.shape)
    min_dist[range(dist.shape[0]), np.argmin(dist, axis=1)] = 1
    return np.sum(dist[min_dist == 1])