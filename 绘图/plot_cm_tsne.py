import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay

# ==========================================
# 全局字体与字号设置
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# 定义各部分字号
TITLE_FONT_SIZE = 22
LABEL_FONT_SIZE = 18
TICK_FONT_SIZE = 16
LEGEND_FONT_SIZE = 14
CM_TEXT_FONT_SIZE = 18


def get_nice_ticks(data_array, nbins=4):
    """
    自动计算动态“顶格”刻度
    """
    data_min, data_max = np.min(data_array), np.max(data_array)
    # 预留 5% 的边界，防止点直接压在坐标轴边框上
    margin = (data_max - data_min) * 0.05

    # 使用 MaxNLocator 自动计算“漂亮”的刻度数字
    locator = MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10])
    ticks = locator.tick_values(data_min - margin, data_max + margin)

    return (ticks[0], ticks[-1]), ticks


def plot_tsne(npz_path, method_name, seed, output_dir):
    print(f"正在处理 t-SNE: {method_name} - Seed {seed}...")
    tsne_data = np.load(npz_path)
    features = tsne_data['features']
    labels = tsne_data['labels']

    # 运行 t-SNE 降维
    tsne = TSNE(n_components=2, perplexity=30, random_state=int(seed))
    features_2d = tsne.fit_transform(features)

    # 核心修改 1：将宽度设为 10，给右侧的图例留出空间，防止主图被挤压
    fig, ax = plt.subplots(figsize=(10, 8))

    unique_labels = np.unique(labels)
    cmap = plt.get_cmap('tab10', len(unique_labels))

    # 绘制散点
    for label in unique_labels:
        mask = np.array(labels) == label
        ax.scatter(features_2d[mask, 0],
                   features_2d[mask, 1],
                   s=50,
                   color=cmap(label),
                   label=f'Type {label + 1}',
                   alpha=0.7,
                   edgecolors='w',
                   linewidths=0.5)

    # ================= 坐标轴顶格与刻度自适应 =================
    xlim, xticks = get_nice_ticks(features_2d[:, 0])
    ylim, yticks = get_nice_ticks(features_2d[:, 1])

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # 设置刻度朝内，且四面都有刻度
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE,
                   direction='in', top=True, bottom=True, left=True, right=True,
                   length=6, width=1)

    # 核心修改 2：强制主画框（Axes）物理长宽比为 1:1 (绝对正方形)
    ax.set_box_aspect(1)
    # ==========================================================

    # 设置标题和轴标签
    ax.set_title('t-SNE Visualization', fontsize=TITLE_FONT_SIZE, pad=15)
    ax.set_xlabel('Dimension 1', fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel('Dimension 2', fontsize=LABEL_FONT_SIZE)

    # 设置图例外置
    ax.legend(title='Fault Types',
              title_fontsize=LEGEND_FONT_SIZE,
              fontsize=LEGEND_FONT_SIZE - 2,
              bbox_to_anchor=(1.03, 1),
              loc='upper left')

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'tsne_{method_name}_seed_{seed}.tif')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_confusion_matrix(npy_path, method_name, seed, output_dir):
    print(f"正在处理 Confusion Matrix: {method_name} - Seed {seed}...")
    cm = np.load(npy_path)

    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制混淆矩阵
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(1, 7))

    # 放大矩阵内部的数字字体
    disp.plot(cmap='Blues', ax=ax, values_format='d',
              text_kw={'fontsize': CM_TEXT_FONT_SIZE, 'fontname': 'Times New Roman'})

    # 设置标题和轴标签
    ax.set_title('Confusion Matrix', fontsize=TITLE_FONT_SIZE, pad=15)
    ax.set_xlabel('Predicted Label', fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel('True Label', fontsize=LABEL_FONT_SIZE)

    # 设置刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)

    # 设置 colorbar 的刻度字体大小
    cbar = ax.images[-1].colorbar
    cbar.ax.tick_params(labelsize=TICK_FONT_SIZE)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'cm_{method_name}_seed_{seed}.tif')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == "__main__":
    # 创建统一的输出文件夹
    output_base_dir = "Paper_Figures_HighRes"
    os.makedirs(output_base_dir, exist_ok=True)

    # 获取当前目录下所有以 _cm 结尾的文件夹
    method_folders = [f for f in os.listdir('.') if os.path.isdir(f) and f.endswith('_cm')]

    if not method_folders:
        print("未找到以 '_cm' 结尾的文件夹，请检查脚本运行目录！")

    for folder in method_folders:
        method_name = folder.replace('_cm', '')

        method_output_dir = os.path.join(output_base_dir, method_name)
        os.makedirs(method_output_dir, exist_ok=True)

        # 1. 处理所有 t-SNE 数据
        tsne_files = glob.glob(os.path.join(folder, 'tsne_data_seed_*.npz'))
        for tsne_file in tsne_files:
            seed = os.path.basename(tsne_file).split('_')[-1].split('.')[0]
            plot_tsne(tsne_file, method_name, seed, method_output_dir)

        # 2. 处理所有混淆矩阵数据
        cm_files = glob.glob(os.path.join(folder, 'confusion_matrix_seed_*.npy'))
        for cm_file in cm_files:
            seed = os.path.basename(cm_file).split('_')[-1].split('.')[0]
            plot_confusion_matrix(cm_file, method_name, seed, method_output_dir)

    print(f"\n所有高清图表已生成完毕，保存在 '{output_base_dir}' 文件夹中！")