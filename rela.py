import numpy as np
from sklearn.neighbors import NearestNeighbors
import math
# Calculate the relative density of each point
class Rela:
    def rela(X):
        #Find the required number of nearest neighbors k
        shape_x = X.shape[0]
        pow_shape = math.pow(shape_x, 0.5)
        k = math.ceil(pow_shape)
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(X)
        neigh1 = neigh.kneighbors(X, return_distance=True)
        a, b = neigh1
        # a represents the distance of k-nearest neighbor samples, b is the index of k-nearest neighbor samples
        
        avg_dis = []
        for i_a in a:
            avg_dis.append((sum(i_a)+0.0001) / (len(i_a) - 1))
        #  avg_dis represents the average of the distances of the k nearest neighbors of all points
        mean = []
        for i_avg in avg_dis:
            mean.append(1 / i_avg)
        # mean represents the reciprocal of the mean distance
        rela_density = []
        for i_mean in mean:
            rela_density.append(i_mean/sum(mean))
        # rela_density represents the proportion of each term of d in the sum, representing the relative density of the sample

        arr_density = np.array(rela_density)
        arr_dis = np.array(avg_dis)
        #arr_cc is a cost-sensitive parameter
        cost = np.sqrt(np.abs(arr_dis))
        list_f = list(np.true_divide(arr_density,cost))
        return list_f

    
