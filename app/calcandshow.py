import json
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def readjson(jsonfile):
    # 读入json文件
    with open(jsonfile, 'r') as f:
        r = json.load(f)
        return r

r = readjson('year_data.json')
#print(type(r))

"""
def getnamemap(r):
    d = {}
    rd = {}
    id = 0
    for k1 in r.keys():  # 年份
        L1 = r[k1]['arr']
"""

def getnamemap(r):
    mp = {}
    rmp = {}
    id = 0
    L1 = r['2006']['arr']
    for d1 in L1:
        mp[d1['name']] = id
        rmp[id] = d1['name']
        id = id + 1
    return mp,rmp

mp,rmp = getnamemap(r)

print(len(mp))
print(len(rmp))
peonum = len(mp)
yearnum = 11

def getRankMatrix():
    R = np.zeros((peonum, yearnum))
    Rv = np.zeros((peonum,yearnum))
    for k1 in r.keys():  # 年份
        L1 = r[k1]['arr']
        for d1 in L1:
            if d1['name'] in mp.keys():
                R[mp[d1['name']]][int(k1) - 2006] = int(d1['mean_rank'])
                Rv[mp[d1['name']]][int(k1) - 2006] = np.sqrt(int(d1['variance']))
    return R,Rv

"""
for i in range(5):
    print(rmp[i])
    for j in range(11):
        print(R[i][j],' ',end='')
    print('')
"""

R,Rv = getRankMatrix()

def showTSNE(R):
    R_tsne = TSNE().fit_transform(R)
    #plt.subplot(121)
    plt.scatter(R_tsne[:, 0], R_tsne[:, 1])
    plt.show()
    return R_tsne

R_tsne = showTSNE(R)
clusternum = 8

def calcKmeans(myR_tsne):
    estimator = KMeans(n_clusters=clusternum)
    estimator.fit(myR_tsne)
    label_pred = estimator.labels_
    #centroids = estimator.cluster_centers_
    #inertia = estimator.inertia_
    return label_pred

label_pred = calcKmeans(R_tsne)
print(label_pred.shape)
#print(label_pred)

def showCluster(myR_tsne,mylabel_pred):
    #plt.subplot(122)
    plt.scatter(myR_tsne[:, 0], myR_tsne[:, 1], c=mylabel_pred)
    plt.show()

showCluster(R_tsne,label_pred)

#R_pca = PCA().fit_transform(R)
#plt.scatter(R_pca[:,0],R_pca[:,1])
#plt.show()

def showTrend(myR,mylabel_pred):
    X = [i for i in range(yearnum)]
    colorlist = ['r', 'g', 'b', 'm', 'k', 'c', 'y', 'peru', 'goldenrod', 'skyblue', 'pink']
    for i in range(peonum):
        if mylabel_pred[i]==0:
            plt.subplot(241)
        elif mylabel_pred[i]==1:
            plt.subplot(242)
        elif mylabel_pred[i] == 2:
            plt.subplot(243)
        elif mylabel_pred[i] == 3:
            plt.subplot(244)
        elif mylabel_pred[i] == 4:
            plt.subplot(245)
        elif mylabel_pred[i] == 5:
            plt.subplot(246)
        elif mylabel_pred[i] == 6:
            plt.subplot(247)
        elif mylabel_pred[i] == 7:
            plt.subplot(248)
        #print(len(myR[i][:yearnum]))
        plt.plot(X,myR[i][:yearnum],c=colorlist[mylabel_pred[i]])
    # plt.figure()
    # plt.gca().invert_yaxis()
    plt.show()

showTrend(R,label_pred)

newR = np.hstack([R,Rv])

print(newR.shape)

newR_tsne = showTSNE(newR)
print(newR_tsne.shape)

newlabel_pred = calcKmeans(newR_tsne)
print(newlabel_pred.shape)
showCluster(newR_tsne,newlabel_pred)
showTrend(newR,newlabel_pred)