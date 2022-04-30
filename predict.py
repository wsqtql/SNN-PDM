from trela import Trela
from snn import SNN
from split import Split
import numpy as np


class Predict:
    def predict(a,X,y):
        #Separating samples from different classes
        sp = Split.split(X,y)
        tnum = len(sp)#Number of classes in the training set
        pre=[]#pre is used to store the prediction results
        sp2 = []#sp2 stores the separation results of each class after snn clustering
        for i1 in range(0, tnum):
            snn = SNN(neighbor_num=10,
                      min_shared_neighbor_proportion=0.5)
            snn.fit(np.array(sp[i1]))
            
            if hasattr(snn, 'labels_'):
                y_pred = snn.labels_.astype(np.int)
            else:
                y_pred = snn.predict(np.array(sp[i1]))
            sp2.append(Split.split(np.array(sp[i1]), y_pred))
        temp = 0
        #Predict each sample in test set a
        for i_a in a:
            all_com=[]#Store the largest calculated result in every class
            c_com=[]#c_com Store the calculation results of the different class clusters for each class
            #cluster_num is the number of clusters in each class，except noise
            cluster_num = []
            for i_t in range(0,len(sp2)):
                sp3 = sp2[i_t]#Assign the separated classes to sp3
                if -1 in sp3.keys():#Noise samples are not added to the calculation
                    cluster_num.append(len(sp3)-1)
                else:
                    cluster_num.append(len(sp3))
                for i2 in range(0,cluster_num[i_t]):
                    #com is the intra-class prediction multiplied by the number of samples within the class (parameter for balanced comparison)
                    com = (Trela.trela(i_a,np.array(sp3[i2]))*len(sp3[i2]))
                    c_com.append(com)
                all_com.append(max(c_com))
            pre.append(np.argmax(all_com))#Obtain the highest predicted value for the corresponding class
        arr_pre = np.array(pre)
        return arr_pre

