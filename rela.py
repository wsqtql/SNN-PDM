import numpy as np
from sklearn.neighbors import NearestNeighbors
import math
# 计算每个点的相对密度
# 新版本
class Rela:
    def rela(X):
        # X=X.reshape(-1,2)
        # print(X)
        a1 = X.shape[0]
        b1 = math.pow(a1, 0.5)
        k = math.ceil(b1)
        neigh = NearestNeighbors(n_neighbors=k)
        # neigh2 = NearestNeighbors(n_neighbors=6)
        # print(X)
        neigh.fit(X)
        # neigh2.fit(X)
        neigh1 = neigh.kneighbors(X, return_distance=True)
        # neigh21 = neigh2.kneighbors(X, return_distance=True)
        a, b = neigh1
        # a为距离，b为下标
        # a2, b2 = neigh21
        c = []
        # c2 = []
        # 计算平均距离10近邻
        # for i_a2 in a2:
        #     c2.append(sum(i_a2) / len(i_a2) - 1)
        #

        for i_a in a:
            # print(i_a)
            c.append((sum(i_a)+0.0001) / (len(i_a) - 1))
        #  c中为所有点的k个近邻的距离的平均值
        # print(c)
        d = []
        for i_c in c:
            d.append(1 / i_c)
        # d为距离平均值的倒数
        # print(d)
        e = []
        for i_d in d:
            # print(type(i_d / sum(d)))
            # arr_c = np.array(c)
            # np.multiply(a,b)
            # print(1,np.array(c).shape)
            # print(2,(i_d/sum(d)).shape)
            # print(3,i_d/sum(d))
            # e.append(np.multiply(i_d / sum(d),arr_c))
            e.append(i_d/sum(d))
        # e为平均值倒数在平均值倒数和中的占比
        # print(1,len(c))
        # print(2,len(e))
        # print(type(e))

        arr_e = np.array(e)
        arr_c = np.array(c)
        arr_cc = np.sqrt(np.abs(arr_c))
        list_f = list(np.true_divide(arr_e,arr_cc))


        # 以下为10近邻的概率密度
        # arr_c2 = np.array(c2)
        # arr_cc2 = np.sqrt(np.abs(arr_c2))
        # list_f = list(np.true_divide(arr_e,arr_cc2))
        return list_f

    