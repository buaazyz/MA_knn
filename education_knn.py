import pandas as pd
import numpy as np
import random as rd
from sklearn import neighbors
#读入训练集
df1 = pd.read_csv('train2.csv')

#读入测试集
df2 = pd.read_csv('test2.csv')


# 验证集的生成
list = []

for i in range(0,80):
    list.append(i)
rd.shuffle(list)

uidlist = df1['id'].loc[0:79]

tlist1=[]
tlist2=[]
tlist3=[]
tlist4=[]
tlist5=[]

for i in range(0,80):
    mod = i%5
    if(mod == 0):
        tlist1.append(uidlist[i])
    elif(mod == 1):
        tlist2.append(uidlist[i])
    elif(mod == 2):
        tlist3.append(uidlist[i])
    elif(mod == 3):
        tlist4.append(uidlist[i])
    else:
        tlist5.append(uidlist[i])



cdf1 = df1[df1['id'].isin(tlist1)]
cdf2 = df1[df1['id'].isin(tlist2)]
cdf3 = df1[df1['id'].isin(tlist3)]
cdf4 = df1[df1['id'].isin(tlist4)]
cdf5 = df1[df1['id'].isin(tlist5)]

for i in range(0,5):
    if(i == 0):
        vdf1 = cdf2.append(cdf3.append(cdf4.append(cdf5)))
    elif(i == 1):
        vdf2 = cdf1.append(cdf3.append(cdf4.append(cdf5)))
    elif(i == 2):
        vdf3 = cdf1.append(cdf2.append(cdf4.append(cdf5)))
    elif(i == 3):
        vdf4 = cdf1.append(cdf3.append(cdf2.append(cdf5)))
    else:
        vdf5 = cdf1.append(cdf3.append(cdf4.append(cdf2)))

#----交叉验证集合生成完毕，cdf1对应vdf1


#定义函数，进行knn计算
#参数vdf代表训练集，cdf代表验证集或者测试集,k代表knn的k值,resname代表文件名
def validation(cdf, vdf, k, resname):
    #建立KNN模型
    neigh = neighbors.KNeighborsClassifier(n_neighbors=k)
    df5 = pd.DataFrame(columns = ['id','month','predict'])

    #按月遍历训练
    for i in range(1,9):
        traindata = vdf[['k1','k2','k3','k4','k5','k6']][vdf['month'] == i].values
        tagdata = vdf[['dismiss']][vdf['month'] == i].values
        
        neigh.fit(traindata,tagdata)
        #喂入
        inputdata = cdf[['k1','k2','k3','k4','k5','k6']][cdf['month'] == i].values
        p = 0;

        #检验    
        for j in inputdata:
            pred = neigh.predict([j])[0]
            pid = cdf.loc[p]['id']
            month = i
            df4 = pd.DataFrame(data=[[pid,month,pred]],columns = ['id','month','predict'])
            df5 = df5.append(df4)
            p = p+1

    df6 = pd.DataFrame(columns=['id','value'])

    df5 = df5.reset_index()
    dic = {}
    for index, row in df5.iterrows():
        dic[row['id']]=0

    for index, row in df5.iterrows():
        pid = row['id']
        month = row['month']
        pre = row['predict']
        value = dic[pid] + pre*month/36
        dic[pid] = value
    
    f = open(resname+str(k)+'.txt','w')
    f.write(str(dic))
    f.close()

#调用函数
validation(df2,df1,5,'test_k')
