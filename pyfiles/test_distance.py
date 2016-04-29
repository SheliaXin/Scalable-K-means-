
import numpy as np
from numpy.testing import assert_almost_equal
from distance_func import distance

def test_non_negativity():
    u = np.random.normal(size=(3,4))
    v = np.random.normal(size=(5,4))
    assert (distance(u, v)>= 0).all()
    
def test_coincidence_when_zero():
    u = np.zeros((3,4))
    v = np.zeros((5,4))
    assert (distance(u, v)==0).all()

def test_coincidence_when_not_zero():
    u = np.random.normal(size=(3,4))
    v = np.random.normal(size=(5,4))
    assert (distance(u, v)!= 0).any()

def test_symmetry():
    u = np.random.normal(size=(3,4))
    v = np.random.normal(size=(5,4))
    assert (distance(u, v)== distance(v, u).T).all()

def test_known1():
    u = np.array([[0,0],[1,1]])
    v = np.array([[0,0],[1,1]])
    dist = np.array([[0,2],[2,0]])
    assert_almost_equal(distance(u, v), dist)
    
def test_known2():
    u = np.array([[0,0,0],[1,1,1],[2,2,2]])
    v = np.array([[1,1,1],[2,2,2],[3,3,3]])
    dist = np.array([[3,12,27],[0,3,12],[3,0,3]])
    assert_almost_equal(distance(u, v), dist)