
import numpy as np
from numpy.testing import assert_almost_equal
from cost_func import cost
from distance_func import distance

def test_non_negativity():
    for i in range(10):
        data = np.random.normal(size=(5,4))
        c = data[np.random.choice(range(4),2),]
        dist = distance(data,c)
        assert cost(dist) >= 0