import scipy.cluster.hierarchy as sch
import numpy as np
import matplotlib.pylab as plt
from sklearn.decomposition import PCA
import os
#删除目录下所有文件
def del_file(path='scipy_hierarchical_clustering/'):
    for i in os.listdir(path):
        path_file = os.path.join(path, i) # 取文件绝对路径
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)
# 距离函数 用于替换 metric='euclidean'，格式 metric=distance
def distance(a, b):
    #距离：需要确定每一个分量的权重，然后求和，欧氏距离等都没有考虑权重问题。致使近期距离影响小，远期影响大，违背事实，建议各分量需要单位化
    #c=(a-b)**2，s为权值向量，dist=np.sum（sc）
    #选择不同的权值向量s，最终通过计算总平均收益，确定具体的数值，总平均收益越大模型越好
    #单位化：各分量减最小值，除最大值，将各分量确定在[0,1]
    #总平均收益：sum(每一类的平均收益×类中元素个数)/总个数
    W = [1,2,3,4,5,6,7,8,9,10,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]#权重
    dist = np.sum((a-b)**2*W)
    return dist

def sch_distPdist_linkage_fcluster(points, t, method='average', metric='euclidean'):
    # 1. 层次聚类
    # 生成点与点之间的距离矩阵,这里用的欧氏距离:
    disMat = sch.distance.pdist(points, metric)
    #print(disMat)
    # 进行层次聚类:
    Z = sch.linkage(disMat, method, metric)
    # # 将层级聚类结果以树状图表示出来并保存为plot_dendrogram.png
    #P = sch.dendrogram(Z)
    #plt.savefig('plot_dendrogram.png')
    # # 根据linkage matrix Z得到聚类结果:
    cluster = sch.fcluster(Z, t, criterion='maxclust')
    #print("Original cluster by hierarchy clustering:\n", cluster)

    # PCA 主成份分析 用于结果的散点图显示
    pca = PCA(2)  # 选取2个主成份
    pca.fit(points)
    low_d = pca.transform(points)  # 降低维度

    plt.ion()
    #plt.figure()
    mark = ['pb', 'or', 'ob', 'og', 'ok', 'oy', 'om', 'oc',
                  'sr', 'sb', 'sg', 'sk', 'sy', 'sm', 'sc',
                  'pr', 'pb', 'pg', 'pk', 'py', 'pm', 'pc',
                  'Dr', 'Db', 'Dg', 'Dk', 'Dy', 'Dm', 'Dc']
    for i in range(len(points)):
        markIndex = int(cluster[i])%28+1  # 为样本指定颜色
        plt.plot(low_d[i, 0], low_d[i, 1], mark[markIndex], markersize=6)
        #plt.text(points[i, 0], points[i, 1], i)
    plt.grid()
    plt.pause(3)
    plt.clf()
    return cluster

def main_4(n = 50, method='complete'):
    del_file(path='scipy_hierarchical_clustering/')  # 分类数可能不同，聚类前需要删除上一次聚类分析中保存的所有类
    for j in range(16):
        L = np.loadtxt('Li_init/L%s.txt' % j)
        v = np.loadtxt('Li_init/v%s.txt' % j)
        f = np.hstack((L, v[:, 1:]))
        k = f.shape[0] // n  # k表示分类的数目，平均50个数据为一类
        cluster = sch_distPdist_linkage_fcluster(f[:, 1:], k, method='complete', metric=distance)  # 'euclidean'
        # print(cluster)
        cluster = cluster.reshape((len(cluster), 1))
        for i in range(1, k + 1):
            ln = np.where(cluster == i)[0]
            Ld = f[ln]
            np.savetxt('scipy_hierarchical_clustering/L%s_%s.txt' % (j, i), Ld)
    print('step_4:聚类')
    print('聚类前需要删除上一次聚类分析中保存的所有类')
    print('每一类平均n个元素：', n)
    print('距离权重W需要在主程序中定义')
    print('类保存在scipy_hierarchical_clustering/L%s_%s.txt')
    print('------')
    return 0
if __name__ == '__main__':
    #W = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 无法写到main_4中
    W = [1, 2, 3, 4, 5, 6, 7,8, 9, 10]
    n = 50      # 每一类平均元素个数
    main_4(n = 20, method='complete')
