import numpy as np
from distance_func import distance
from kmeans_func import KMeans

def test_label():
    for i in range(10):
        data = np.random.normal(size=(50,2))
        k = 3
        centroids = data[np.random.choice(range(data.shape[0]), k, replace=False),:]
        label = model = KMeans(data, k, centroids)["Labels"]
        assert max(label) == k-1 and len(label)==data.shape[0]