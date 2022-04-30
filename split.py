import numpy as np
# Split the data set X according to the value of the label y，this method will be used before and after clustering
class Split:
    def split(X,y):
        a={}
        for t in range(0,(len(np.unique(y)))):
            b = []
            for i in range(len(X)):
                if t == y[i]:
                    b.append(X[i])
            if len(b) != 0:
                a.update({t: b})
        return a
