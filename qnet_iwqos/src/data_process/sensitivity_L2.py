# This code is part of LINKEQ.
#
# (C) Copyright LINKE 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#
# -*- coding: utf-8 -*-
# @Time     : 2024/2/25 2:16
# @Author   : HFLASH @ LINKE
# @File     : sensitivity_L2.py
# @Software : PyCharm
# from datas import small_data_no_remote as A
# from datas import small_data_L2_remote as B
# from datas import small_data_L1_remote as A
# from datas import small_data_L1L2_remote as B

# from datas import large_data_no_remote as A
# from datas import large_data_L2_remote as B
# from datas import large_data_L1_remote as A
# from datas import large_data_L1L2_remote as B

import matplotlib.pyplot as plt

def draw_sensitivity_L2(bw, strategy):
    if strategy == 'L2':
        from datas import small_data_no_remote as A
        from datas import small_data_L2_remote as B
    if strategy == 'L1L2':
        from datas import small_data_L1_remote as A
        from datas import small_data_L1L2_remote as B
    xticks = []
    for key, value in A.items():
        if key[0] == bw:
            percentages_A = {key: value[3] / (value[1] + value[3]) * 100}
            xticks.append(key)
    for key, value in B.items():
        if key[0] == bw:
            percentages_B = {key: value[3] / (value[1] + value[3]) * 100}

    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    index = range(5)
    y_min = min(min(percentages_A.values()), min(percentages_B.values())) * 0.8  # 略低于最小值
    y_max = max(max(percentages_A.values()), max(percentages_B.values())) * 1.2  # 略高于最大值
    ax.set_ylim([y_min, y_max])

    bar1 = ax.bar(index, list(percentages_A.values()), bar_width, label='Random', color='blue')
    bar2 = ax.bar([p + bar_width for p in index], list(percentages_B.values()), bar_width, label='Oldest-first',
                  color='red')

    small_circuit = ['adr4_197', 'cm42a_207', 'cm82a_208', 'rd53_251', 'z4_268']
    # 添加图例和标签
    # ax.set_xlabel('Circuits')
    ax.set_ylabel(r'$\gamma$ (%)', fontsize=24)
    ax.set_title('')
    ax.set_xticks([p + bar_width / 2 for p in index])
    ax.set_xticklabels(small_circuit, fontsize=24)
    ax.legend(fontsize=20)

    # 显示图表
    plt.xticks(rotation=45)  # 旋转x轴标签以便阅读
    plt.tick_params(labelsize=20)
    plt.tight_layout()  # 自动调整子图参数,使之填充整个图像区域
    plt.show()
    # fig.savefig('./sensitivity_L2_bw3.pdf')
    fig.savefig('./sensitivity_fig/L2opt/sensitivity_%s_bw_%s.pdf'%(bw, strategy))


if __name__ == '__main__':
    bws = ['1', '3', '5']
    strategies = ['L1L2', 'L2']
    for bw in bws:
        for strategy in strategies:
            draw_sensitivity_L2(bw, strategy)