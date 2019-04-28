#原始数据处理，加入索引，加入延迟序列
import numpy as np
np.set_printoptions(threshold= 1000000000000)
#去除成交量如周期
def remove_cycle(f):
    # 去除日周期影响：除以每个时间点的平均成交量
    time = [0.00, 0.17, 0.33, 0.50, 0.67, 0.83]
    for i in time:
        index_i = np.where(f[:, 0] == i)[0]
        f[index_i, -1] = f[index_i, -1] / np.mean(f[index_i, -1])
    return f

#data 转换
def dataReshape(X, dtw= 10):
    dtwL = np.ones([len(X), dtw+1])
    for i in range(dtw+1):
        dtwL[dtw-i:, i] = X[:len(X)-(dtw-i)]
    return dtwL

def main_1(filePath = 'GOLD_H4.txt', dtw = 10):
    #载入表，去除成交量日周期影响
    f = np.loadtxt(filePath)
    f_1 = remove_cycle(f)
    #索引列
    index = np.arange(0, f.shape[0], 1).reshape((f.shape[0], 1))
    #收盘价列
    f_close = f[:, -2]
    close_11 = dataReshape(f_close, dtw)  # dtwL=[10天前的数据，9天前的数据，...,昨天的，今天的]共11列数据，最后一列为预测数据，倒数第二列为真实数据
    close_index_11 = np.hstack((index, close_11))
    #成交量列
    f_vol = f_1[:, -1]
    vol_11 = dataReshape(f_vol, dtw)
    vol_index_11 = np.hstack((index, vol_11))

    np.savetxt('raw_data_procession/close_index_11.txt', close_index_11)
    np.savetxt('raw_data_procession/vol_index_11.txt', vol_index_11)

    print('step_1:原始数据处理')
    print('原始数据文件：', filePath)
    print('选取前n天的数据用于聚类：', dtw)
    print('处理后数据的列向量代表的意义：索引，Bar[-9],...,Bar,预测列')
    print('收盘价数据保存在：raw_data_procession/close_index_11.txt')
    print('成交量数据保存在：raw_data_procession/vol_index_11.txt')
    print('------')
    return 0

if __name__=="__main__":
    filePath = 'GOLD_H4.txt'   #载入原始数据表
    dtw = 10        #选取前10天的数据用于分析
    main_1(filePath, dtw)

