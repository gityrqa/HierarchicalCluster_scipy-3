import numpy as np
import os
def file_name(file_dir):        #获取目录flie_dir下所有子文件的文件名
    for root, dirs, files in os.walk(file_dir):
        #print(root)  # 当前目录路径
        #print(dirs)  # 当前路径下所有子目录
        #print(files)  # 当前路径下所有非目录子文件
        return files
def main_5():
    files = file_name('scipy_hierarchical_clustering')
    raw_data_procession = np.loadtxt('raw_data_procession/close_index_11.txt')  # 使用这个表中的数据计算收益
    profit = np.zeros((len(files), 2))
    ergodic = np.zeros((1, 4))
    t = 0
    for i in files:
        # print(i)
        f = np.loadtxt('scipy_hierarchical_clustering/' + i)  # 仅利用类中的索引
        # 若分类中只有个一个数据，则表现为一维的，需要转为2维数组
        if f.ndim == 1:
            f = f.reshape((1, len(f)))
        index_f = f[:, 0].astype('int')  # 本身是浮点数类型的需要转为int类型，才可以作为索引
        # print(index_f)
        f_r = raw_data_procession[index_f, :]
        profit_f = (f_r[:, 11] - f_r[:, 10]) / f_r[:, 10]
        profit_f_per = np.mean(profit_f)
        print(i, len(profit_f), profit_f_per)
        profit[t, :] = [len(profit_f), profit_f_per]
        t = t + 1
        #
        ergodic_per = np.zeros((len(index_f), 4))
        ergodic_per[:, 0] = index_f  # 索引
        ergodic_per[:, 1] = len(index_f)  # 类中元素个数
        ergodic_per[:, 2] = profit_f_per  # 类平均盈亏
        ergodic_per[:, 3] = profit_f  # 各元素实际盈亏
        ergodic = np.vstack((ergodic, ergodic_per))
    # print(profit)
    ergodic = np.delete(ergodic, 0, axis=0)
    ergodic = ergodic[ergodic[:, 0].argsort()]  # 排序 根据某一列排序

    np.savetxt('profit_data/profit.txt', profit)  # 每一类的元素数与平均收益表
    np.savetxt('profit_data/ergodic.txt', ergodic)  # 每个元素的情况

    with open('profit_data/files.txt', 'w') as f:  # 每一类的文件名
        f.write(str(files))
        f.close()
    # with open('model_assession/profit_dict.txt', 'r') as f:   #读取
    #     a = f.read()
    #     a = eval(a)
    #     print(a)
    #     print(type(a))
    #     f.close()
    print('step_5:收益汇总')
    print('每一类只用到了索引，通过索引在步骤1生成的表中查找并计算')
    print('保存每一类的文件名到：profit_data/files.txt.txt')
    print('保存每一类的元素数和平均收益：profit_data/profit.txt')
    print('保存每一个元素的情况[序号，所在类元素数，平均收益，元素本身收益]：profit_data/ergodic.txt')
    print('------')
if __name__ == '__main__':
    main_5()




