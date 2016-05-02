# STA663-Final-Project: Scalable K-Means

### Team Members: Xin Xu, Fu Wen

This is the final project for course STA663. In this project, we implement the `K-means||` algorithm in Python following Bahmani's paper "Scalable k-means++" and speed up using `Cython` and paralleled using `multiprocessing`. 

We apply it to the simulated `GAUSSMIXTURE` dataset and the `SPAM` dataset from UC Irvine Machine Learning repository. In this part, we compare the misclassification rate (for `GAUSSMIXTURE` dataset), clustering cost and runtime of the `k-means||` algorithm with these of random initialization and the `k-means++`. In the end, we inplement `k-means||` algorithm in Spark using `pyspark.millib`. 

From the implementation on the simulated dataset and the real-world dataset, the `k-means||` and `k-means++` find a better initial centroids than random in most cases, which leads to a better final clustering performance. Also, `k-means||` runs faster than `k-means++`, since it runs a fewer number of rounds and been speed up in the parallel implementation.

However, the clustering cost of `k-means||` is not that stable. We think it might be caused by the problem in the last step. We firstly tried reclustering the multiple weighted centroids in $C$ into $k$ clusters. Since the initial center for the reclustering is picked randomly , it will have the same problem as using random initialization. Then, referring to some online resources, we tried sampling $k$ final centroids from the weighted centroids in $C$ in the last step. But we think it might also get several centroids in one large cluster and lead to an unstable result. 
