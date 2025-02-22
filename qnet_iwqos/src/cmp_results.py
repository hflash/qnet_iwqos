import re
import pandas as pd
import numpy as np

if __name__ == "__main__":

    method1 = 'greedy'
    method2 = 'random'
    # trivial
    # greedy
    # hungarian
    # algebraic_connectivity_ random_
    # log1 = f'/home/normaluser/hflash/qnet_iwqos/test_data/alg_test/allocation/random_{method1}_global_hetero_random_{method1}.csv'
    # log2 = f'/home/normaluser/hflash/qnet_iwqos/test_data/alg_test/allocation/random_{method2}_global_hetero_random_{method2}.csv'
    log1 = f'/home/normaluser/hflash/qnet_iwqos/test_data/alg_test/allocation/algebraic_connectivity_{method1}_global_hetero_random_{method1}.csv'
    log2 = f'/home/normaluser/hflash/qnet_iwqos/test_data/alg_test/allocation/algebraic_connectivity_{method2}_global_hetero_random_{method2}.csv'
    # log1 = f'/home/normaluser/hflash/qnet_iwqos/test_data/alg_test/allocation/random_{method1}_baseline_hetero_random_{method1}.csv'
    # log2 = f'/home/normaluser/hflash/qnet_iwqos/test_data/alg_test/allocation/random_{method2}_baseline_hetero_random_{method2}.csv'
    # log1 = f'/home/normaluser/hflash/qnet_iwqos/test_data/alg_test/allocation/algebraic_connectivity_{method1}_baseline_hetero_random_{method1}.csv'
    # log2 = f'/home/normaluser/hflash/qnet_iwqos/test_data/alg_test/allocation/algebraic_connectivity_{method2}_baseline_hetero_random_{method2}.csv'
    data1 = pd.read_csv(log1,index_col=0)
    data2 = pd.read_csv(log2,index_col=0)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 500)
    cmp = (data1-data2)/data1
    data1_Ecost = data1['entanglement cost'] +  data1['discard entanglements']
    data2_Ecost = data2['entanglement cost'] +  data2['discard entanglements']
    dist_opt = (data1['time cost']-data2['time cost'])/data1['opt']
    # mean_var_1 = data1['var'] / data1['mean']
    # mean_var_2 = data2['var'] / data1['mean']
    # cmp = (data1 - data2)
    varbymean = data1['cost_var']/data1['cost_mean']
    df = pd.concat([data1['time cost'], data2['time cost'], data1['cost_mean'], data2['cost_var'], cmp['time cost'], dist_opt, varbymean], axis=1, keys=[method1, method2, 'cost_mean', 'cost_var','compare', 'distopt', 'varbymean'])

    print(df)
    print(np.mean(cmp['time cost']))
    # print(np.mean(cmp['time cost']))
    print(np.mean(dist_opt))
    