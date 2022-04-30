from trela import Trela
from snn import SNN
from split import Split
import numpy as np


class Predict2:
    def predict2(a,X,y):
        sp = Split.split(X,y)
        tnum = len(sp)
        b=[]
        sp2 = []
        for i1 in range(0, tnum):
            snn = SNN(neighbor_num=10,
                      min_shared_neighbor_proportion=0.5)
            snn.fit(np.array(sp[i1]))

            if hasattr(snn, 'labels_'):
                y_pred = snn.labels_.astype(np.int)
            else:
                y_pred = snn.predict(np.array(sp[i1]))
            sp2.append(Split.split(np.array(sp[i1]), y_pred))
            # sp2为字典组成的数组
            # print('sp2的长度')
            # 增加的操作如下
        temp = 0
        for i_a in a:
            d=[]
            temp = temp+1
            c=[]
            tnum2 = []
            for i_t in range(0,len(sp2)):
                sp3 = sp2[i_t]
                if -1 in sp3.keys():
                    tnum2.append(len(sp3)-1)
                else:
                    tnum2.append(len(sp3))
                for i2 in range(0,tnum2[i_t]):
                    e = (Trela.trela(i_a,np.array(sp3[i2]))*len(sp3[i2]))
                    c.append(e)
                d.append(max(c))
            b.append(np.argmax(d))
        arr_b = np.array(b)
        return arr_b

