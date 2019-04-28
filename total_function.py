from step_1_raw_data_procession import main_1
from step_2_ma_class import main_2
from step_3_Li_init import main_3
from step_4_scipy_hierarchical_clustering import main_4
from step_5_profit_data import main_5
from step_6_model_assession import main_6

if __name__=="__main__":
    usecols = 4     #选择获取的数据：第5列
    dtw = 10        #选取前10天的数据用于分析
    main_1( 'GOLD.txt', usecols, dtw)

    ma_1 = 50
    ma_2 = 90
    k = 100  # 去掉前100个数据
    main_2(filePath='raw_data_procession/gold_intex_-10_0.txt', ma_1=ma_1, ma_2=ma_2, k=k)

    main_3(filePath='ma_class/L%s.txt')
    #W = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]更改权重需要到step_4中进行
    n = 50  # 每一类平均元素个数
    main_4(n, method='complete')
    main_5()

    n_1 = 30  # 类中元素数不低于50个
    n_2 = 0.0008  # 类的平均利润不低于0.001
    main_6(n_1, n_2)