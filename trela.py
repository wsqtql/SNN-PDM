import numpy as np
from rela import Rela

# 新版本
class Trela:
    def trela(a,X):
        Y = X.copy()
        # print(Y.shape)
        # print(type(a))
        # print(a.shape)
        # Z = np.array([Y,a])
        # Z = np.append([Y],[a],axis=0)
        t = a.shape[0]
        b = a.reshape(1,t)
        Z = np.concatenate((Y,b),axis=0)
        # print(Z)
        # print(Z.shape)
        ans = Rela.rela(Z)
        final = ans[-1]
        return final


