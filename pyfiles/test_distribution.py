
import numpy as np
from numpy.testing import assert_almost_equal
from cost_func import cost
from distance_func import distance
from distribution_func import distribution

def test_non_negative():
    data = np.random.normal(size=(20,4))
    c = data[np.random.choice(range(4),4),]
    dist = distance(data,c)
    c = cost(dist)
    p = distribution(dist,c)
    assert (p>=0).all()
    
def test_sum_to_one():
    data = np.random.normal(size=(20,4))
    c = data[np.random.choice(range(4),4),]
    dist = distance(data,c)
    c = cost(dist)
    p = distribution(dist,c)
    assert_almost_equal(np.sum(p),1)