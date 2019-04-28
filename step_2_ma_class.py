#ma平滑，对倒数第二列数据（当天数据），根据他的二阶差分进行分类
#最后一列数据为预测序列
import numpy as np

#SMA  计算简单移动平均
def sma(X,ma_list):
    b = np.zeros(len(X))
    for i in range(0, len(X)):#i取0到len(X)-1
        if i < ma_list:
            b[i] = sum(X[0:i+1])/(i+1)#[0:2]取0,1元素相加
        else:
            b[i] = sum(X[i+1-ma_list:i+1])/ma_list#要取到当天的需要写为[n:i+1]
    return b
#对data 按照二阶斜率分类
def maSlope(X, maS= 50):
    smaS = sma(X, maS-10)
    smaS = sma(smaS, 10)
    s = X.shape[0]
    Y = np.zeros([s, 2])
    Y[1:, 0] = np.diff(smaS, 1)* 1000
    Y[2:, 1] = np.diff(smaS, 2)* 1000
    #print(Y)
    Ylabel = np.zeros([s, 2])
    for i in range(0, s):
        if Y[i, 0] >= 0:
            Ylabel[i, 0] = 0
        else:
            Ylabel[i, 0] = 1
        if Y[i, 1] >= 0:
            Ylabel[i, 1] = 0
        else:
            Ylabel[i, 1] = 1
    #print(Ylabel.shape)
    return Ylabel
# 2进制标签转为10进制
def translation_2to10(label_2):
    label_10 = label_2[:,0]*8+label_2[:,1]*4+label_2[:,2]*2+label_2[:,3]
    label_10 = label_10.reshape((len(label_10), 1))
    return label_10

def main_2(ma_1 = 50, ma_2 = 90, k=100):
    #通过均线分类
    filePath = 'raw_data_procession/close_index_11.txt'
    a = np.loadtxt(filePath)
    f = a[:, 10]  # 倒数第二列，找到当天数据列
    label_2 = np.hstack((maSlope(f, ma_1), maSlope(f, ma_2)))  # 数组合并
    label_10 = translation_2to10(label_2)  #
    #close数据处理
    label_10_index = np.hstack((a, label_10))[k:, :]  # 合并为13列数组
    np.savetxt('ma_class/label_10_index.txt', label_10_index)
    for i in range(16):
        li = np.where(label_10_index[:, -1] == i)[0]
        Ld = label_10_index[li]
        np.savetxt('ma_class/L%s.txt' % i, Ld)

    #vol数据处理
    vol_Path = 'raw_data_procession/vol_index_11.txt'
    vol = np.loadtxt(vol_Path)
    label_10_index = np.hstack((vol, label_10))[k:, :]  # 合并为13列数组
    for i in range(16):
        li = np.where(label_10_index[:, -1] == i)[0]
        Ld = label_10_index[li]
        np.savetxt('ma_class/v%s.txt' % i, Ld)
    print('step_2:用两条移动平均线的2阶差分将数据分为16类')
    print('去掉前k个数据：', k)
    print('增加一列：分类序号列')
    print('处理后数据总表保存在：ma_class/label_10_index.txt')
    print('------')

if __name__=="__main__":
    ma_1 = 50
    ma_2 = 90
    k = 100     #去掉前100个数据
    main_2(ma_1=50, ma_2=90, k=100)
