import numpy as np
def sample_new(data,distribution,l):
    """ Sample new centers
    Parameters:
       data                       n*d
       distribution               n*1
       l                          the number of new centers to sample
    Returns:
       sample_new                 
    """
    return data[np.random.choice(range(len(distribution)),l,p=distribution),:]