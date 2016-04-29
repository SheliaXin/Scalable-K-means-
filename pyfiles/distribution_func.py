import numpy as np
def distribution(dist,cost):
    """ Calculate the distribution to sample new centers
    Parameters:
       dist                      distance matrix between data and current centroids
       cost                      the cost of data with respect to the current centroids
    Returns:
       distribution 
    """
    return np.min(dist, axis=1)/cost