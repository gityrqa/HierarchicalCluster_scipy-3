import numpy as np

f=np.loadtxt("GOLD_H4.txt")
#数据编号,生成一个带编号的数组：index_f
index = np.array(np.arange(f.shape[0])).reshape((f.shape[0], 1))
index_f = np.hstack((index, f))
#去除日周期影响：除以每个时间点的平均成交量
time = [0.00, 0.17, 0.33, 0.50, 0.67, 0.83]
for i in time:
    index_0 = np.where(index_f[:, 1] == i)[0]
    index_f[index_0, -1] = index_f[index_0, -1]/np.mean(index_f[index_0, -1])
    print(index_f[index_0, -1])
print(index_f)
f_1 = index_f