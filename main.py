"""
接受时间集成序列矩阵
首先利用dtw算出各行之间的距离
再用MDS进行降维，K-Means进行聚类
对每一类用不同颜色画出其随时间的变化趋势
观察结果
输入：一个列表
[
[  [],[],[],......,[]  ],
[  [],[],[],......,[]  ],
......
[  [],[],[],......,[]  ]
]
为了方便后面绘制趋势图像，还需要直到每行的始末时间
可以作为另一个输入
输出：绘制的相关图像，和每一行所属的类别
"""
"""
相关导入
"""
import math
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

"""
dtw的测试：

"""

# the distance between two node in the time series
# 计算序列两个节点之间的距离
# 如果节点是向量，则计算向量之间的欧式距离
# 如果节点是数值，则计算数值差的绝对值
def calcdist(x,y):
    examplelist = [1, 2, 3, 4, 5]
    # if the node is a list
    if type(x) == type(examplelist):
        length = len(x)
        sum = 0
        for i in range(length):
            sum = sum + (x[i]-y[i])*(x[i]-y[i])
        return math.sqrt(sum)
    else:# the node is just a number
        return abs(x-y)

# DP return the distance
# 动态规划计算出两个不一定等长序列之间的距离
# 输入，两个序列，以及序列节点之间距离的计算函数
# 返回，两个序列之间的距离
def dynamicTimeWarp(seqA, seqB, d ):
    # create the cost matrix
    # 构建花费矩阵
    numRows, numCols = len(seqA), len(seqB)
    cost = [[0 for _ in range(numCols)] for _ in range(numRows)]

    # record father
    # 记录父亲
    fa = [[(0,0) for _ in range(numCols)] for _ in range(numRows)]

    # initialize the first row and column
    # 初始化第一行和第一列
    cost[0][0] = d(seqA[0], seqB[0])
    for i in range(1, numRows):
        cost[i][0] = cost[i - 1][0] + d(seqA[i], seqB[0])
        fa[i][0] = (i-1,0)

    for j in range(1, numCols):
        cost[0][j] = cost[0][j - 1] + d(seqA[0], seqB[j])
        fa[0][j] = (0,j-1)

    # fill in the rest of the matrix
    # 填充剩余的矩阵
    for i in range(1, numRows):
        for j in range(1, numCols):
            # choices = cost[i - 1][j], cost[i][j - 1], cost[i - 1][j - 1]
            # cost[i][j] = min(choices) + d(seqA[i], seqB[j])
            if cost[i-1][j] < cost[i][j-1]:
                if cost[i-1][j] < cost[i-1][j-1]:
                    cost[i][j] = cost[i-1][j] + d(seqA[i], seqB[j])
                    fa[i][j] = (i-1,j)
                else:
                    cost[i][j] = cost[i - 1][j-1] + d(seqA[i], seqB[j])
                    fa[i][j] = (i-1,j-1)
            else:
                if cost[i][j-1] < cost[i-1][j-1]:
                    cost[i][j] = cost[i][j-1] + d(seqA[i], seqB[j])
                    fa[i][j] = (i,j-1)
                else:
                    cost[i][j] = cost[i - 1][j - 1] + d(seqA[i], seqB[j])
                    fa[i][j] = (i-1,j-1)

    # if want to show the cost matrix, uncomment the following four sentences ctrl+/
    # for row in cost:
    #     for entry in row:
    #         print ("%03d" % entry,end="")
    #     print("")

    # if want to show the path, uncomment the following ten sentences
    # path = []
    # i = numRows - 1
    # j = numCols - 1
    # path.append((i,j))
    # while i!=0 or j!=0:
    #    i,j=fa[i][j]
    #    path.append((i,j))
    # for cord in path[::-1]:
    #     print(cord, ' ', end="")
    # print("")
    return cost[-1][-1]

"""
测试dtw
seqA = [0,0,0,3,6,13,25,22,7,2,1,0,0,0,0,0,0]
seqB = [0,0,0,0,0,0,4,5,12,24,23,8,3,1,0,0]
# 返回7
seqA = [[0,0,0,0],[0,0,0,0],[1,1,1,1],[2,2,2,2],[3,3,3,3],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
seqB = [[0,0,0,0],[2,2,2,2],[3,3,3,3],[4,4,4,4],[0,0,0,0]]
# 返回4
dist = dynamicTimeWarp(seqA,seqB,calcdist)
print(dist)
"""

"""
计算出距离矩阵
假设输入一共有n行，那么
可以得到n*n的一个距离矩阵，
第(i,j)格代表第i行所代表的元素与第j行所代表的元素之间的距离
距离使用dtw进行计算
"""

# 读入json，返回列表
def readinputlist(jsonfile):
    with open(jsonfile,'r') as f:
        r = json.load(f)
        return r

# 计算距离矩阵
def calcdistmat(r):
    length = len(r) # 行数
    D = np.array([ [0 for i in range(length)] for j in range(length) ])
    for i in range(length):
        for j in range(length):
            D[i][j] = dynamicTimeWarp(r[i], r[j], calcdist)
    np.save('distmat.npy',D)
    return D

"""
MDS降维部分
输入：读入高维空间中的距离矩阵，低维空间的维数k
输出：高维向量对应的低维向量
"""

# 如果是npy格式，则使用这个读入npy，传成array
def loaddistmat(f):
    A = np.load(f)
    return A

"""
读入高维距离矩阵n*n，和降至的维度mydim
输出MDS降维后的距离矩阵，一般为n*2
"""
def mds(Dist,mydim):
    n = len(Dist)
    print(n)
    B = np.zeros((n,n))
    disti2 = np.array([0] * n)
    print(disti2.shape)
    distj2 = np.array([0] * n)
    print(distj2.shape)
    for x in range(n):
        disti2[x] = np.mean([Dist[x][j] for j in range(n)])  # 距离矩阵每行的平均值
        distj2[x] = np.mean([Dist[i][x] for i in range(n)])  # 距离矩阵每列的平均值
    print(disti2.shape)
    print(distj2.shape)
    distij2 = np.mean([Dist[i][j] for i in range(n) for j in range(n)])  # 距离矩阵的平均值
    print(distij2.shape)
    for i in range(n):
        for j in range(n):
            B[i][j] = -0.5 * (Dist[i][j] - disti2[i] - distj2[j] + distij2)  # 投影后的距离矩阵
    w, v = np.linalg.eig(B)  # 计算矩阵的特征值及特征向量
    # w the eigenvalues each repeated according to its multiplicity
    # v the normalized(united "length") eigenvectors
    # such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i]
    v = v.transpose()
    U = [{'eVal': w[i], 'eVec': v[i]} for i in range(n)]
    U.sort(key=lambda obj: obj.get('eVal'), reverse=True)  # 按特征值从大到小排序
    k = mydim  # 降至mydim维，一般取2，3维
    w = np.array([0] * k)  # k*1 all zero
    v = np.zeros((k, n))  # k*n all zero
    print(w)
    print(w.shape)
    print(v.shape)
    print(v)
    for i in range(k):
        w[i] = U[i].get('eVal') ** 0.5  # 特征值开根号
        v[i] = U[i].get('eVec')  # 特征向量
    # n*k * k*k = n*k
    ans = np.dot(v.transpose(), np.diag(w)) # 矩阵相乘
    print(ans.shape)
    # if you want to compare the distance matrix between the high dim or the low dim, uncomment the following eleven sentences
    # print(ans.shape)
    # print(ans)
    # ans_dist = np.zeros((n, n))
    # for i in range(n):
    #     ans_str = ""
    #     for j in range(n):
    #         ans_dist[i][j] = sum((ix - jx) ** 2 for ix, jx in zip(ans[i], ans[j]))  # 计算投影后的距离矩阵
    # print("Orign dis[][] is :")
    # print(Dist)
    # print("MDS dis[][] is :")
    # print(ans_dist)
    return ans

# 画出MDS降维后的结果
# 输入：MDS根据特征值分解后得到的对应的低维的点n*lowdim
def plotmds(X):
    plt.scatter(X[:,0],X[:,1])
    plt.show()

# 进行KMeans聚类
# 输入：低维的点的坐标，聚类数目
# 输出：每个点对应的类别标签
def calcKmeans(mycoord,clusternum):
    estimator = KMeans(n_clusters=clusternum)
    estimator.fit(mycoord)
    label_pred = estimator.labels_
    print(label_pred.shape)
    # centroids = estimator.cluster_centers_
    # inertia = estimator.inertia_
    return label_pred

# 用不同颜色展示KMeans聚类后的结果
def showcluster(mycoord,mylabel_pred):
    plt.scatter(mycoord[:,0],mycoord[:,1],c=mylabel_pred)
    plt.show()


# 输入：原始的时序排名集成数据列表，某一个实体所属类别，始末时间，聚类数目
# 输出：将不同类别的排名变化趋势画出来
def display(r,mylabel,st,ed,clusternum):
    # 颜色查找表
    colorlist = ['r', 'g', 'b', 'm', 'k', 'c', 'y', 'peru', 'goldenrod', 'skyblue', 'pink']
    num = len(r) # 参与排名的实体数
    for k in range(num):
        rowlist = r[k] # 某一行
        length1 = len(rowlist)  # 时间节点数
        tmplist = []
        # 对于多属性的时序排名，取各个属性的平均值作为当前时间节点的排名
        for i in range(length1):
            tmpsum = 0
            length2 = len(rowlist[i])  # 属性数
            for j in range(length2):
                tmpsum = tmpsum + rowlist[i][j]
            tmplist.append(float(tmpsum/length2))
        xval = [i for i in range(st[i],ed[i]+1)]
        # 暂时只分成了4或8个聚类，如有需要可以更改，以适应不同的聚类数
        if clusternum == 4:
            if mylabel[k]==0:
                plt.subplot(221)
            elif mylabel[k]==1:
                plt.subplot(222)
            elif mylabel[k]==2:
                plt.subplot(223)
            elif mylabel[k]==3:
                plt.subplot(224)
        elif clusternum == 8:
            if mylabel[k]==0:
                plt.subplot(241)
            elif mylabel[k]==1:
                plt.subplot(242)
            elif mylabel[k]==2:
                plt.subplot(243)
            elif mylabel[k]==3:
                plt.subplot(244)
            elif mylabel[k]==4:
                plt.subplot(245)
            elif mylabel[k]==5:
                plt.subplot(246)
            elif mylabel[k]==6:
                plt.subplot(247)
            elif mylabel[k]==7:
                plt.subplot(248)
        plt.plot(xval, tmplist, c=colorlist[mylabel[k]])
    plt.show()