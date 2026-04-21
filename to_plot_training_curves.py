import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# ================= 参数设置 =================
CSV_PATHS = ["wsst.csv","cwt.csv", "HHT_emd.csv", "vmd1.csv", "ceemdan.csv" , "waveform.csv"]  # 对应的CSV文件路径
DATASET_NAMES = ["WSST","CWT", "HHT", "VMD_HT", "CEEMDAN_HT","1D Wavefrom"]  # 对应的数据集名称
SAVE_PATH = "training_curves_comparison.tif"  # 图片保存路径
COLORS = sns.color_palette("husl", n_colors=len(CSV_PATHS))  # 使用seaborn调色板

# ============================================

# 设置全局字体为Times New Roman
plt.rcParams.update({
    'font.size': 16,
    'font.family': 'Times New Roman',  # 主要修改点
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'figure.titlesize': 20,
    # 确保数学公式字体也同步修改
    'mathtext.default': 'regular',
    'mathtext.fontset': 'stix'
})

# 创建画布
sns.set_style("whitegrid")
fig, axs = plt.subplots(2, 2, figsize=(16, 10))
# fig.suptitle('Training Curves Comparison', y=1.02)

# 循环处理每个数据集
for idx, (csv_path, dataset_name) in enumerate(zip(CSV_PATHS, DATASET_NAMES)):
    df = pd.read_csv(csv_path)

    # 绘制训练曲线（保持原有绘图逻辑）
    axs[0, 0].plot(df.epoch, df.train_loss, color=COLORS[idx], linewidth=2, alpha=0.9, label=dataset_name)
    axs[0, 1].plot(df.epoch, df.val_loss, color=COLORS[idx], linewidth=2, alpha=0.9)
    axs[1, 0].plot(df.epoch, df.train_acc, color=COLORS[idx], linewidth=2, alpha=0.9)
    axs[1, 1].plot(df.epoch, df.val_acc, color=COLORS[idx], linewidth=2, alpha=0.9)

# 统一设置坐标轴格式
titles = ['Training Loss', 'Validation Loss', 'Training Accuracy', 'Validation Accuracy']
ylabels = ['Loss', 'Loss', 'Accuracy', 'Accuracy']

for ax, title, ylabel in zip(axs.flat, titles, ylabels):
    ax.set_title(title, fontname='Times New Roman')  # 显式设置子图标题字体
    ax.set_xlabel('Epoch', fontname='Times New Roman')
    ax.set_ylabel(ylabel, fontname='Times New Roman')
    ax.set_xlim(1, df.epoch.max())
    ax.grid(True, alpha=0.3)

    # 强制设置刻度标签字体
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')

    # 百分比格式设置
    if 'Accuracy' in title:
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

# 设置图例字体
handles = [plt.Line2D([0], [0], color=COLORS[i], lw=3) for i in range(len(CSV_PATHS))]
fig.legend(handles, DATASET_NAMES,
           loc='lower center',
           ncol=len(CSV_PATHS),
           bbox_to_anchor=(0.5, -0.05),
           prop={'family': 'Times New Roman'})  # 显式设置图例字体

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
plt.show()
