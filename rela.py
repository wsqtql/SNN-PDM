import numpy as np
from sklearn.neighbors import NearestNeighbors
import math
# Calculate the relative density of each point
class Rela:
    def rela(X):
        #Find the required number of nearest neighbors k
        a1 = X.shape[0]
        b1 = math.pow(a1, 0.5)
        k = math.ceil(b1)
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(X)
        neigh1 = neigh.kneighbors(X, return_distance=True)
        a, b = neigh1
        # a represents the distance of k-nearest neighbor samples, b is the index of k-nearest neighbor samples
        
        c = []
        for i_a in a:
            c.append((sum(i_a)+0.0001) / (len(i_a) - 1))
        #  c represents the average of the distances of the k nearest neighbors of all points
        d = []
        for i_c in c:
            d.append(1 / i_c)
        # d represents the reciprocal of the mean distance
        e = []
        for i_d in d:
            e.append(i_d/sum(d))
        # e represents the proportion of each term of d in the sum, representing the relative density of the sample

        arr_e = np.array(e)
        arr_c = np.array(c)
        #arr_cc is a cost-sensitive parameter
        arr_cc = np.sqrt(np.abs(arr_c))
        list_f = list(np.true_divide(arr_e,arr_cc))
        return list_f

    
