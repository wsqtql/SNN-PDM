from trela import Trela
from snn import SNN
from split import Split
import numpy as np


class Predict2:
    def predict2(a,X,y):
        sp = Split.split(X,y)
        # print(sp.keys())
        tnum = len(sp)
        # print(sp.keys())
        # print('tnum=')
        # print(tnum)
        # 字典的不同类的个数错了
        b=[]
        # global tnum2
        # tnum2=[]
        sp2 = []
        for i1 in range(0, tnum):
        # for i1 in range(-1, 2, 2):
            snn = SNN(neighbor_num=10,
                      min_shared_neighbor_proportion=0.5)
            # print(sp[i1])
            snn.fit(np.array(sp[i1]))

            if hasattr(snn, 'labels_'):
                y_pred = snn.labels_.astype(np.int)
            else:
                y_pred = snn.predict(np.array(sp[i1]))
            sp2.append(Split.split(np.array(sp[i1]), y_pred))
            # sp2为字典组成的数组
            # print('sp2的长度')
            # 增加的操作如下

            # print(len(sp2))
            # if -1 in sp2.keys():
            #     tnum2.append(len(sp2)-1)
            # else:
            #     tnum2.append(len(sp2))


            # print(tnum2)
            # tnum2为每一类的聚类个数
            # print(sp2)
            # tnum2 = (len(sp2))
            # if -1 in sp2.keys():
            #     tnum2 = tnum2 - 1
            # else:
            #     tnum2 = tnum2
            # print(tnum2)
        temp = 0
        for i_a in a:
            # print(i_a)
            d=[]
            temp = temp+1
            # print(temp)
            # print(tnum2)
            # for i1 in range(0,tnum):
            #     snn = SNN(neighbor_num=20,
            #             min_shared_neighbor_proportion=0.6)
            #     # print(sp[i1])
            #     snn.fit(np.array(sp[i1]))
            #
            #     if hasattr(snn, 'labels_'):
            #         y_pred = snn.labels_.astype(np.int)
            #     else:
            #         y_pred = snn.predict(np.array(sp[i1]))
            #     sp2 = Split.split(np.array(sp[i1]),y_pred)
            #     # print(sp2)
            #     tnum2 = (len(sp2))
            #     if -1 in sp2.keys():
            #         tnum2 = tnum2 - 1
            #     else:
            #         tnum2 = tnum2
            c=[]
            # tnum2 = (len(sp2))
            # if -1 in sp2.keys():
            #     tnum2 = tnum2 - 1
            # else:
            #     tnum2 = tnum2
            tnum2 = []
            for i_t in range(0,len(sp2)):
                # print(tnum2[i_t])
                # print(len(sp2))
                sp3 = sp2[i_t]
                if -1 in sp3.keys():
                    tnum2.append(len(sp3)-1)
                    # target = (len(sp3) - 1)
                    # avg = []
                    # for r in range(target):
                    #     mid = np.array(sp3[r])
                    #     list_mid = list(mid.ravel())
                    #     avg.append(list_mid)
                    # avg_arr = np.array(avg)
                    # for litter in sp3[-1]:
                    #     distances = np.sqrt(np.sum(np.array(litter - avg_arr) ** 2, axis=1))
                    #     # np.argmin(distances)
                    #     min_index = np.argmin(distances)
                    #     sp3[min_index].append(litter)



                else:
                    tnum2.append(len(sp3))
                # print(tnum2[i_t])
                for i2 in range(0,tnum2[i_t]):
                    e = (Trela.trela(i_a,np.array(sp3[i2]))*len(sp3[i2]))
                    # print(e)
                    c.append(e)
                    # print(c)
                d.append(max(c))
                # print(d)
            b.append(np.argmax(d))
        arr_b = np.array(b)
        # print(len(arr_b))
        return arr_b

        # sp = Split.split(X,y_pred)
        # print(len(sp))
        # tnum = len(sp)
        # if -1 in sp.keys():
        #     tnum = tnum - 1
        # else:
        #     tnum = tnum
        #
        # b=[]
        # d = []
        # for i1 in range(0,tnum):
        #     # print(sp[i1])
        #     d.append(len(sp[i1]))
        #     # print(len(d))
        # arr_dd = np.array(d)
        # print(arr_dd)
        # # print(len(arr_dd))
        # for i_a in a:
        #     c = []
        #     # print(i_a)
        #     # print(type(c))
        #     for i in range(0,tnum):
        #         # print(Trela.trela(i_a,np.array(sp[i])))
        #         c.append(Trela.trela(i_a,np.array(sp[i])))
        #         # print(np.array(sp[i]))
        #     # print(c)
        #     arr_aa = np.array(c)
        #     # print(arr_aa)
        #     # print(len(arr_aa))
        #     final = arr_aa*arr_dd
        #     # print(type(final))
        #     # print(final)
        #     maxindex = np.argmax(final)
        #     # print(maxindex)
        #     b.append(maxindex)
        #     # print(final)
        #     # print(len(final))
        #     # print(type(final))
        #
        #
        # arr_bb = np.array(b)
        # return arr_bb
