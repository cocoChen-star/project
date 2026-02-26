import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

# 设置 Times New Roman 字体
times_new_roman = font_manager.FontProperties(family='Times New Roman')
plt.rcParams['font.family'] = 'Times New Roman'

# 只隐藏坐标原点处 (0,0) 的刻度线（保留数字）
def hide_zero_cross_tick_lines(ax):
    for tick in ax.xaxis.get_major_ticks():
        if abs(tick.get_loc()) < 1e-8:
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            break
    for tick in ax.yaxis.get_major_ticks():
        if abs(tick.get_loc()) < 1e-8:
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            break

# 设置坐标轴数字字体
def set_tick_label_font(ax, font, size):
    for label in ax.get_xticklabels():
        label.set_fontproperties(font)
        label.set_fontsize(size)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font)
        label.set_fontsize(size)

# 绘制 PR 曲线
def plot_PR():
    pr_csv_dict = {
        'YOLOv11n': r'runs\detect\val15\PR_curve.csv',
        'YOLOv11n+A': r'runs\detect\val20\PR_curve.csv',
        'YOLOv11n+B': r'runs\detect\val30\PR_curve.csv',
        'YOLOv11n+A+B': r'runs\detect\val50\PR_curve.csv',
    }

    fig, ax = plt.subplots(figsize=(6, 6), tight_layout=True)

    for modelname in pr_csv_dict:
        res_path = pr_csv_dict[modelname]
        x = pd.read_csv(res_path, usecols=[1]).values.ravel()
        y = pd.read_csv(res_path, usecols=[3]).values.ravel()
        ax.plot(x, y, label=modelname, linewidth=2)

    ax.set_xlabel('Recall', fontsize=24, fontproperties=times_new_roman)
    ax.set_ylabel('Precision', fontsize=24, fontproperties=times_new_roman)

    ax.set_xticks([round(i * 0.2, 1) for i in range(6)])
    ax.set_yticks([round(i * 0.2, 1) for i in range(6)])

    set_tick_label_font(ax, times_new_roman, 18)
    hide_zero_cross_tick_lines(ax)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.legend(loc='lower left', prop=times_new_roman, fontsize=20)

    fig.savefig("pr.png", dpi=600)
    plt.show()

# 绘制 F1 曲线
def plot_F1():
    f1_csv_dict = {
        'YOLOv11n': r'runs\detect\val3\F1_curve.csv',
    }

    fig, ax = plt.subplots(figsize=(6, 6), tight_layout=True)

    for modelname in f1_csv_dict:
        res_path = f1_csv_dict[modelname]
        x = pd.read_csv(res_path, usecols=[1]).values.ravel()
        y = pd.read_csv(res_path, usecols=[3]).values.ravel()
        ax.plot(x, y, label=modelname, linewidth=2)

    ax.set_xlabel('Confidence', fontsize=24, fontproperties=times_new_roman)
    ax.set_ylabel('F1', fontsize=24, fontproperties=times_new_roman)

    ax.set_xticks([round(i * 0.2, 1) for i in range(6)])
    ax.set_yticks([round(i * 0.2, 1) for i in range(6)])

    set_tick_label_font(ax, times_new_roman, 18)
    hide_zero_cross_tick_lines(ax)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.legend(loc='lower left', prop=times_new_roman, fontsize=20)

    fig.savefig("F1.png", dpi=500)
    plt.show()

# 主函数入口
if __name__ == '__main__':
    plot_PR()
    # plot_F1()
