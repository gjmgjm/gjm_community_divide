#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from xlrd import open_workbook
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300 #分辨率

def draw(i):
    data = open_workbook('result.xlsx')
    table = data.sheets()[0]
    x = table.col_values(0)
    x1 = x[4:8]
    print(x1)
    y = table.col_values(i)
    y1 = np.array(y[0:4])
    print(y1)
    y2 = np.array(y[4:8])
    print(y2)
    y3 = np.array(y[8:12])
    print(y3)
    plt.ylim(0, 1.0)
    plt.title('用户频繁访问位置变化')
    plt.plot(x1, y1, ".-", color='green', label='random disturb', linewidth=2)
    plt.plot(x1, y2, ".-", color='red', label='community division', linewidth=2)
    plt.plot(x1, y3, ".-", color='blue', label='K-anonymous', linewidth=2)
    plt.legend()  # 显示图例
    plt.grid(axis="y")
    plt.xlabel("k-values")
    plt.savefig(str(i) +"common_loc_change.png")
    plt.close()


if __name__ == "__main__":
    for i in range(6):
        i = i+1
        draw(i)