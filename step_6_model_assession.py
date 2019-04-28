import numpy as np
import matplotlib.pylab as plt
import ch
ch.set_ch()
# 累计求和函数
def cumsum(profit):
    profit_add = np.zeros(shape=(len(profit)))
    for i in range(len(profit)):
        profit_add[i] = np.sum(profit[:i+1])
    return profit_add
def ergodic_profit(a,n_1=50, n_2=0.001):
    b = np.zeros((a.shape[0], 1))
    for i in range(a.shape[0]):
        if a[i, 2] * a[i, 3] >= 0:
            if a[i, 1] > n_1 and np.abs(a[i, 2]) > n_2:
                b[i, 0] = np.abs(a[i, 3])
            else:
                b[i, 0] = 0
        else:
            if a[i, 1] > n_1 and np.abs(a[i, 2]) > n_2:
                b[i, 0] = -np.abs(a[i, 3])
            else:
                b[i, 0] = 0
    c = cumsum(b)
    plt.figure()
    plt.plot(c)
    plt.show()

def main_6(n_1 = 30, n_2 = 0.001):
    profit = np.loadtxt('profit_data/profit.txt')  # 每一类的元素数与平均收益表
    ergodic = np.loadtxt('profit_data/ergodic.txt')  # 每个元素的情况
    #n_1 = 30  # 类中元素数不低于50个
    #n_2 = 0.0008  # 类的平均利润不低于0.001
    plt.ion()
    # 总平均收益、总元素个数、分类数
    profit_t = np.sum(np.abs(profit[:, 1]) * profit[:, 0]) / np.sum(profit[:, 0])
    print('总分类数：', profit.shape[0])
    print('总利润：', np.sum(np.abs(profit[:, 1]) * profit[:, 0]))
    print('总平均利润：', profit_t)
    print('总元素个数：', np.sum(profit[:, 0]))

    # 元素数大于n的类的获利情况：类数、平均收益、总元素个数
    s = 0  # 元素个数大于n的类的总收益
    s_t = 0  # 元素个数大于n的类的元素总数
    s_s = 0  # 元素个数大于n的类的个数
    for i in range(profit.shape[0]):
        if profit[i, 0] > n_1 and np.abs(profit[i, 1]) > n_2:
            s = s + profit[i, 0] * np.abs(profit[i, 1])
            s_t = s_t + profit[i, 0]
            # print(s, s_t)
            s_s = s_s + 1
    print('元素个数大于%s,收益大于%s的类的个数：' % (n_1, n_2), s_s)
    print('元素个数大于%s,收益大于%s的类的总利润：' % (n_1, n_2), s)
    print('元素个数大于%s,收益大于%s的类的总平均利润：' % (n_1, n_2), s / s_t)
    print('元素个数大于%s,收益大于%s的类的元素总数：' % (n_1, n_2), s_t)
    # 作历史情况图
    ergodic_profit(ergodic, n_1, n_2)

    #
    with open('profit_data/files.txt', 'r') as f:  # 读取文件名
        a = eval(f.read())
        f.close()
    plt.figure()
    for i in range(profit.shape[0]):
        if profit[i, 0] > n_1 and np.abs(profit[i, 1]) > n_2:
            data = np.loadtxt('scipy_hierarchical_clustering/' + a[i])
            # print(data)
            # plt.figure(a[i])
            for j in range(data.shape[0]):
                plt.plot(data[j, 1:], 'r')
            plt.title(u'元素总数：%s' % np.int(profit[i, 0]))
            plt.pause(3)
            plt.clf()
            plt.show()
    return 0
if __name__ == '__main__':
    #n_1 = 30  # 类中元素数不低于50个
    #n_2 = 0.0005  # 类的平均利润不低于0.001
    main_6(n_1 = 50, n_2 = 0.0005)