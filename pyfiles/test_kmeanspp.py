import numpy as np
from distance_func import distance
from cost_func import cost
from distribution_func import distribution
from sample_new_func import sample_new
from kmeanspp_func import KMeansPlusPlus
from scalablekmeanspp_func import ScalableKMeansPlusPlus

def test_length():
    data = np.random.normal(size=(2000,2))
    k = 3
    l = 5
    ini1 = KMeansPlusPlus(data, k)
    ini2 = ScalableKMeansPlusPlus(data, k, l)
    assert len(ini1)==k and len(ini2)==k

def test_length():
    data = np.random.normal(size=(2000,2))
    k = 3
    l = 5
    ini1 = KMeansPlusPlus(data, k)
    ini2 = ScalableKMeansPlusPlus(data, k, l)
    check1 = [i in data for i in ini1]
    check2 = [i in data for i in ini1]
    assert all(check1) and all(check2)