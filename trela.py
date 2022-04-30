import numpy as np
from rela import Rela
#Compute the relative density of the test sample a on the sample set X，return the relative density value
class Trela:
    def trela(a,X):
        Y = X.copy()
        t = a.shape[0]
        b = a.reshape(1,t)
        Z = np.concatenate((Y,b),axis=0)
        ans = Rela.rela(Z)
        final = ans[-1]
        return final


