import numpy as np
from distance_func import distance
from cost_func import cost
from distribution_func import distribution
from sample_new_func import sample_new
from get_weight_func import get_weight

def ScalableKMeansPlusPlus(data, k, l):
    
    """ Apply the KMeans|| clustering algorithm
    
    Parameters:
      data                        ndarrays data 
      k                           number of cluster
      l                           number of point sampled in each iteration
    
    Returns:
      "Centroids"                 the final centroids finded by KMeans  
      
    """
    
    centroids = data[np.random.choice(range(data.shape[0]),1), :]
    
    #Calculate the iteration time
    dist = distance(data,centroids)
    iter = int(np.log(cost(dist)))
    
    for i in range(iter):
        #Get the distance between data and centroids
        dist = distance(data, centroids)
        
        #Calculate the cost of data with respect to the centroids
        norm_const = cost(dist)
        
        #Calculate the distribution for sampling l new centers
        p = distribution(dist,norm_const)
        
        #Sample the l new centers and append them to the original ones
        centroids = np.r_[centroids, sample_new(data,p,l)]
    

    ## reduce k*l to k using KMeans++ 
    dist = distance(data, centroids)
    weights = get_weight(dist,centroids)
    
    return centroids[np.random.choice(range(len(weights)),k,p=weights),:]