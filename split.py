import numpy as np
# 根据y去切分X
class Split:
    def split(X,y):
        a={}
        for t in range(0,(len(np.unique(y)))):
        # for t in range(-1, 2, 2):
            b = []
            for i in range(len(X)):
                if t == y[i]:
                    b.append(X[i])
            if len(b) != 0:
                a.update({t: b})
        return a
