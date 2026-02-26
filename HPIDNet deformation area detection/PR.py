import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

# 设置 Times New Roman 字体
times_new_roman = font_manager.FontProperties(family='Times New Roman')

# 全局设置字体
plt.rcParams['font.family'] = 'Times New Roman'

# 绘制 PR 曲线
def plot_PR():
    pr_csv_dict = {
        'Group 1': r'runs\detect\val10\PR_curve.csv',
        'Group 2': r'runs\detect\val15\PR_curve.csv',
        'Group 3': r'runs\detect\val20\PR_curve.csv',
        'Group 4': r'runs\detect\val50\PR_curve.csv',
        'Group 5': r'runs\detect\val30\PR_curve.csv',
        'Group 6': r'runs\detect\val150\PR_curve.csv',


    }

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)  # 正方形图像

    for modelname in pr_csv_dict:
        res_path = pr_csv_dict[modelname]
        x = pd.read_csv(res_path, usecols=[1]).values.ravel()
        data = pd.read_csv(res_path, usecols=[3]).values.ravel()
        ax.plot(x, data, label=modelname, linewidth=2)

    # 设置坐标轴标签（新罗马 + 字体大）
    ax.set_xlabel('Recall', fontsize=24)
    ax.set_ylabel('Precision', fontsize=24)

    # 设置坐标轴刻度字体大小和字体
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_xticks([round(i * 0.2, 1) for i in range(6)])  # 横轴步长 0.2
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 设置图例（增大字体，使用 Times New Roman，放左下）
    legend = ax.legend(loc='lower left', prop=times_new_roman, fontsize=20)
    for label in legend.get_texts():
        label.set_fontsize(20)

    fig.savefig("pr.png", dpi=600)
    plt.show()

# 绘制 F1 曲线
def plot_F1():
    f1_csv_dict = {
        'Group 1': r'runs\detect\val20\F1_curve.csv',
        'Group 2': r'runs\detect\val40\F1_curve.csv',
        'Group 3': r'runs\detect\val50\F1_curve.csv',
        'Group 4': r'runs\detect\val150\F1_curve.csv',
        'Group 5': r'runs\detect\val35\F1_curve.csv',
        'Group 6': r'runs\detect\val250\F1_curve.csv',
    }

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)

    for modelname in f1_csv_dict:
        res_path = f1_csv_dict[modelname]
        x = pd.read_csv(res_path, usecols=[1]).values.ravel()
        data = pd.read_csv(res_path, usecols=[3]).values.ravel()
        ax.plot(x, data, label=modelname, linewidth=2)

    ax.set_xlabel('Confidence', fontsize=24)
    ax.set_ylabel('F1', fontsize=24)

    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 设置图例（增大字体，使用 Times New Roman，放左下）
    legend = ax.legend(loc='lower left', prop=times_new_roman, fontsize=20)
    for label in legend.get_texts():
        label.set_fontsize(20)

    fig.savefig("F1.png", dpi=600)
    plt.show()

if __name__ == '__main__':
    # plot_PR()   # 绘制 PR
    plot_F1()   # 如需绘制 F1，取消注释
