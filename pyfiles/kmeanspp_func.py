import numpy as np
from distance_func import distance
from cost_func import cost
from distribution_func import distribution
from sample_new_func import sample_new

def KMeansPlusPlus(data, k):
    
    """ Apply the KMeans++ clustering algorithm
    
    Parameters:
      data                        ndarrays data 
      k                           number of cluster
    
    Returns:
      "Centroids"                 the final centroids finded by KMeans  
      
    """
    
    #Initialize the first centroid
    centroids = data[np.random.choice(data.shape[0],1),:]
    
    while centroids.shape[0] < k :
                
        #Get the distance between data and centroids
        dist = distance(data, centroids)
        
        #Calculate the cost of data with respect to the centroids
        norm_const = cost(dist)
        
        #Calculate the distribution for sampling a new center
        p = distribution(dist,norm_const)
        
        #Sample the new center and append it to the original ones
        centroids = np.r_[centroids, sample_new(data,p,1)]
    
    return centroids