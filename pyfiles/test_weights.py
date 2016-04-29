
import numpy as np
from numpy.testing import assert_almost_equal
from cost_func import cost
from distance_func import distance
from get_weight_func import get_weight

def test_non_negative():
    data = np.random.normal(size=(20,4))
    centroids = data[np.random.choice(range(4),4),]
    dist = distance(data,centroids)
    w = get_weight(dist,centroids)
    assert (w>=0).all()
    
def test_sum_to_one():
    data = np.random.normal(size=(20,4))
    centroids = data[np.random.choice(range(4),4),]
    dist = distance(data,centroids)
    w = get_weight(dist,centroids)
    assert_almost_equal(np.sum(w),1)