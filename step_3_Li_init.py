# 给ma_class/L...中的数据进行归一化和单位化
import numpy as np

def main_3(filePath='L%s.txt'):
    for j in range(16):
        f = np.loadtxt('ma_class/'+filePath % j)
        f_1_10 = f[:, 1:11]
        f_1_10_mean_1 = np.mean(f_1_10, axis=1)#按行平均
        f_1_10_mean_1 = f_1_10_mean_1.reshape((len(f_1_10_mean_1),1))
        #print('f_1_10_mean_1.shape:', f_1_10_mean_1)
        f_1_10_mean = f_1_10/f_1_10_mean_1#除以每行的平均值
        f_1_10_mean_min = np.min(f_1_10_mean, axis=0)
        f_1_10_mean_max = np.max(f_1_10_mean, axis=0)
        f_1_10_mean_min_1=f_1_10_mean-f_1_10_mean_min#减每列的最小值
        f_1_10_mean_min_max = f_1_10_mean_min_1/(f_1_10_mean_max-f_1_10_mean_min)#除以(每列的最大值-最小值）
        f_index = f[:,0]
        f_index = f_index.reshape((len(f_index), 1))
        f_init = np.hstack((f_index, f_1_10_mean_min_max))
        #print(f_init.shape)
        np.savetxt('Li_init/'+filePath % j, f_init)#11列，不含预测类和分类列
    print('step_3:均线分类数据归一化单位化')
    print('每行除行平均，每列减去列最小值，除(列最大值-列最小值);使数据位于[0,1]')
    print("只取前11列数据，不包含预测列、分类序号列")
    print('数据保存于：Li_init/'+filePath)
    print('------')

    return 0
if __name__ == '__main__':
    main_3(filePath='L%s.txt')      #filePath = 'ma_class/'+'L%s.txt'
    main_3(filePath='v%s.txt')