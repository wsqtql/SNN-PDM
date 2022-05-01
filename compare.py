from predict2 import Predict2
import numpy as np
from split import Split
from sklearn.metrics import jaccard_score
from sklearn.model_selection import train_test_split
import pandas as pd
import scipy.io as io
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.metrics import f1_score
from gaussiannb import GaussianNB2
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import SMOTE
from pdm import PDM
from snngnb import SNNGNB
from measure import Measure



path = '/home/S2_WSQ/project/dataok/ok_satimage1.csv'
# path = 'E:\Desktop\dataok\ok_vowel.csv'
df = pd.read_csv(path)
        # if list =='mat1.csv' or list =='mat2.csv' or list =='mat3.csv' or list =='mat4.csv' or list =='mat5.csv':
        #     df = df.drop('Unnamed: 0',axis=1) #删除第一列pandas产生的unnamed列
        #     df.columns = range(len(df.columns))#df.columns是长度为自己宽度的一个一维数组
        #     df.index=range(len(df))#df.index是大小为df长度的一个一维数组
        #     df[len(df.columns)-1]=df[len(df.columns)-1].replace(0,-1)#把标签0改成-1
        # else:
df.columns = range(len(df.columns))
df[len(df.columns) - 1] = df[len(df.columns) - 1].replace('negative', -1)
df[len(df.columns) - 1] = df[len(df.columns) - 1].replace('positive', 1)
mm = MinMaxScaler()  # 归一化
original_data = mm.fit_transform(df.iloc[:, 0:len(df.columns) - 1])
original_data = pd.DataFrame(original_data)
original_data[len(df.columns) - 1] = df[len(df.columns) - 1]
original_data=np.array(original_data)
dataMat=original_data[:,0:-1]#原始数据集
# print(dataMat)
labelMat=original_data[:,-1]#原始标签集
# print(labelMat)
# labelMat2 = [int(x == 1) for x in labelMat]

# path = 'E:\Desktop\mdatabase/banknote.mat'
# original_data = io.loadmat(path)  # 读取mat文件
# original_data = pd.DataFrame(original_data['data'])
# original_data=np.array(original_data)
# dataMat=original_data[:,0:-1]#原始数据集
# labelMat=original_data[:,-1]#原始标签集
#
# min_max_scaler = preprocessing.MinMaxScaler()
# dataMat2 = min_max_scaler.fit_transform(dataMat)
# labelMat2 = [int(x == 1) for x in labelMat]

# path = '/home/S2_WSQ/project/mdatabase/wilt.mat'
# original_data = io.loadmat(path)  # 读取mat文件
# original_data = pd.DataFrame(original_data['data'])
# original_data=np.array(original_data)
# dataMat=original_data[:,0:-1]#原始数据集
# labelMat=original_data[:,-1]#原始标签集
#
# min_max_scaler = preprocessing.MinMaxScaler()
# dataMat2 = min_max_scaler.fit_transform(dataMat)
# labelMat2 = [int(x == 1) for x in labelMat]
print(path)

r1 = []
r2 = []
r3 = []
r4 = []
r5 = []
r6 = []
r7 = []
r8 = []
g1 =[]
g2 =[]
g3 =[]
g4 =[]
g5 =[]
g6 =[]
g7 =[]
g8 =[]

ep = 0
for epoch in range(10):
    ep = ep+1
    print('epoch',ep)
    X_train,X_test,y_train,y_test = train_test_split(dataMat,labelMat,test_size=0.2)
    # print(y_test)
    # print(y_test)
    # X1,y1 = make_blobs(n_samples=1000, centers=7, n_features=2, cluster_std=5,random_state=0)
    # print(len(X_train))


    y_predict=Predict2.predict2(X_test,X_train,y_train)
    # print(y_predict)
    # print(y_test)
    # print(X_train.shape)
    # print(y_predict)
    # jac1 = jaccard_score(y_test, y_predict, average='micro')
    jac1 = f1_score(y_test, y_predict, average='weighted')

    r1.append(jac1)
    g1.append(Measure.gmean(y_test,y_predict))
    # print("snn方法")
    # print(jac1)

    gnb = GaussianNB().fit(X_train,y_train)
    y_pre1 = gnb.predict(X_test)
    jac2 = f1_score(y_test, y_pre1, average='weighted')
    r2.append(jac2)
    g2.append(Measure.gmean(y_test, y_pre1))
    # print("gnb方法")
    # print(jac2)


    gnb1 = GaussianNB2()
    gnb1.fit(X_train,y_train)
    y_pre2 = gnb1.predict(X_test)
    # print('y_pre2',y_pre2)
    # print('y',y_test)
    jac3 = f1_score(y_test, y_pre2, average='weighted')
    r3.append(jac3)
    g3.append(Measure.gmean(y_test, y_pre2))
    # print('IR_gnb')
    # print(jac3)

    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    gnb3 = GaussianNB().fit(X_resampled,y_resampled)
    y_pre3 = gnb3.predict(X_test)
    jac4 = f1_score(y_test, y_pre3, average='weighted')
    r4.append(jac4)
    g4.append(Measure.gmean(y_test, y_pre3))
    # print("ros")
    # print(jac4)

    rus = ClusterCentroids(random_state=0)
    X_resampled1, y_resampled1 = rus.fit_resample(X_train, y_train)
    gnb4 = GaussianNB().fit(X_resampled1,y_resampled1)
    y_pre4 = gnb4.predict(X_test)
    jac5 = f1_score(y_test, y_pre4, average='weighted')
    r5.append(jac5)
    g5.append(Measure.gmean(y_test, y_pre4))
    # print("rus")
    # print(jac5)

    smote = SMOTE(random_state=0)
    X_resampled2, y_resampled2 = smote.fit_resample(X_train, y_train)
    gnb5 = GaussianNB().fit(X_resampled2,y_resampled2)
    y_pre5 = gnb5.predict(X_test)
    jac6 = f1_score(y_test, y_pre5, average='weighted')
    r6.append(jac6)
    g6.append(Measure.gmean(y_test, y_pre5))
    # print("smote")
    # print(jac6)


    y_pre6 = PDM.pdm(X_test,X_train,y_train)
    jac7 = f1_score(y_test, y_pre6, average='weighted')
    r7.append(jac7)
    g7.append(Measure.gmean(y_test, y_pre6))
    # print("PDM")
    # print(jac7)

    y_pre7 = SNNGNB.snngnb(X_test,X_train,y_train)
    jac8 = f1_score(y_test, y_pre7, average='weighted')
    r8.append(jac8)
    g8.append(Measure.gmean(y_test, y_pre7))
    # print("SNNGNB")
    # print(jac8)
    # with open('./export.txt', 'a') as f:
    #     f.write(path + '\n')
    # np.savetxt('./export.txt',(y_test,y_predict,y_pre1,y_pre2,y_pre3,y_pre4,y_pre5,y_pre6,y_pre7))

score1 = np.array(r1).mean()
score2 = np.array(r2).mean()
score3 = np.array(r3).mean()
score4 = np.array(r4).mean()
score5 = np.array(r5).mean()
score6 = np.array(r6).mean()
score7 = np.array(r7).mean()
score8 = np.array(r8).mean()
gmean1 = np.array(g1).mean()
gmean2 = np.array(g2).mean()
gmean3 = np.array(g3).mean()
gmean4 = np.array(g4).mean()
gmean5 = np.array(g5).mean()
gmean6 = np.array(g6).mean()
gmean7 = np.array(g7).mean()
gmean8 = np.array(g8).mean()
print('snn',score1,gmean1)
print('gnb',score2,gmean2)
print('ir_gnb',score3,gmean3)
print('ros',score4,gmean4)
print('rus',score5,gmean5)
print('smote',score6,gmean6)
print('snngnb',score8,gmean8)
print('pdm',score7,gmean7)
# print('snn',score1)
# print('gnb',score2)
# print('ir_gnb',score3)
# print('ros',score4)
# print('rus',score5)
# print('smote',score6)
# print('snngnb',score8)
# print('pdm',score7)
# info1 = ['snn',str(score1),str(gmean1)]
# info2 = ['gnb',str(score2),str(gmean2)]
# info3 = ['ir_gnb',str(score3),str(gmean3)]
# info4 = ['ros',str(score4),str(gmean4)]
# info5 = ['rus',str(score5),str(gmean5)]
# info6 = ['smote',str(score6),str(gmean6)]
# info7 = ['snngnb',str(score8),str(gmean8)]
# info8 = ['pdm',str(score7),str(gmean7)]
# answer = [str(info1),str(info2),str(info3),str(info4),str(info5),str(info6),str(info7),str(info8)]
# with open('./export.txt','a') as f:
#     f.write(path+'\n')
#     for i in answer:
#         f.write(i+'\n')



# for i in len(info):
#     if i % 2 == 0
#         f.ri