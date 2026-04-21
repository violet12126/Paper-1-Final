import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 填入你提供的真实数据 (10次测试的准确率 %, 乘以了100)
# ==========================================
data_wsst = np.array([98.00, 97.39, 96.42, 97.34, 98.00, 97.68, 97.45, 97.67, 96.44, 97.36])
data_cwt = np.array([83.05, 96.33, 97.18, 97.18, 96.61, 97.18, 96.61, 95.48, 96.61, 96.89])
data_hht = np.array([67.51, 56.78, 61.86, 74.29, 87.01, 75.14, 73.73, 58.76, 74.01, 88.70])
data_vmd = np.array([81.92, 76.27, 86.72, 87.57, 79.94, 72.60, 77.40, 87.01, 85.88, 89.27])
data_ceemdan = np.array([68.93, 59.32, 58.19, 57.91, 61.58, 57.91, 53.39, 62.15, 65.25, 58.76])
data_waveform = np.array([88.07, 85.42, 88.26, 89.02, 88.26, 85.80, 89.96, 89.02, 90.15, 86.93])
# 将数据组织成 DataFrame 方便绘图
df = pd.DataFrame({
    'WSST': data_wsst,
    'CWT': data_cwt,
    'VMD-HT': data_vmd,
    'HHT': data_hht,
    'CEEMDAN-HT': data_ceemdan,
    '1D Waveform': data_waveform
})

# 转换成长格式 (Long format)
df_melted = df.melt(var_name='Method', value_name='Accuracy')

# ==========================================
# 2. 统计学分析 (非参数检验)
# ==========================================
print("=== 统计学非参数检验结果 ===")

# A. Kruskal-Wallis H-test (多组整体比较)
stat_kw, p_kw = stats.kruskal(df['WSST'], df['CWT'], df['VMD-HT'], df['HHT'], df['CEEMDAN-HT'])
print(f"1. Kruskal-Wallis 整体检验:\n   H-statistic = {stat_kw:.4f}, p-value = {p_kw:.4e}")
if p_kw < 0.05:
    print("   -> 结论: 5种方法的准确率整体上存在显著的统计学差异 (p < 0.05)。\n")
else:
    print("   -> 结论: 整体无显著差异。\n")

# B. Mann-Whitney U test (WSST 与次优方法 CWT 的两两比较)
# alternative='greater' 表示单侧检验：WSST 是否显著大于 CWT
stat_mw, p_mw = stats.mannwhitneyu(df['WSST'], df['CWT'], alternative='greater')
print(f"2. Mann-Whitney U 检验 (WSST vs CWT):")
print(f"   U-statistic = {stat_mw:.4f}, p-value = {p_mw:.4f}")
if p_mw < 0.05:
    print("   -> 结论: WSST 的准确率显著高于 CWT (p < 0.05)。\n")
else:
    print("   -> 结论: WSST 与 CWT 差异不显著 (p >= 0.05)。\n")

# ==========================================
# 3. 绘制学术箱线图
# ==========================================
#
plt.rcParams['font.family'] = 'Times New Roman'
sns.set_theme(style="ticks", font="Times New Roman")

plt.figure(figsize=(10, 6))

# 画箱线图
ax = sns.boxplot(x='Method', y='Accuracy', data=df_melted,
                 palette='Set2', width=0.5, boxprops=dict(alpha=0.8))

# 叠加散点图 (展示这10次实验的具体分布，显得数据真实可信)
sns.stripplot(x='Method', y='Accuracy', data=df_melted,
              color='black', alpha=0.6, jitter=True, size=5)

# 坐标轴和标题设置
plt.title('Distribution of Diagnostic Accuracy (10 Independent Trials)', fontsize=16, fontweight='bold', pad=15)
plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
plt.xlabel('Time-Frequency Method', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 在图上标注显著性差异
# WSST 与 CWT 的对比
if p_mw < 0.05:
    # 获取柱子中心的 x 坐标
    x1, x2 = 0, 1  # WSST 是第 0 个，CWT 是第 1 个
    y, h, col = df['WSST'].max() + 1.0, 0.5, 'k'

    # 画线和星号
    plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)

    # 决定标几个星号 (p<0.05: *, p<0.01: **, p<0.001: ***)
    stars = "***" if p_mw < 0.001 else "p < 0.01" if p_mw < 0.01 else "*"
    plt.text((x1 + x2) * .5, y + h, stars, ha='center', va='bottom', color=col, fontsize=16)

# 添加网格线并优化边框
plt.grid(axis='y', linestyle='--', alpha=0.7)
sns.despine()

# 保存高质量图片 (300 dpi 满足期刊要求)
plt.tight_layout()
plt.savefig('Statistical_Boxplot_Accuracy_RealData.tif', dpi=300)
print("-> 箱线图已成功保存为 'Statistical_Boxplot_Accuracy_RealData.png'")
plt.show()