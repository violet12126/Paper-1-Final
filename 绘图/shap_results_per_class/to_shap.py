import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import shap
import matplotlib.pyplot as plt
from PIL import Image
import random


# ================= 1. 配置与随机种子 =================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


GLOBAL_SEED = 42
seed_everything(GLOBAL_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = [f'Type {i + 1}' for i in range(6)]


# ================= 2. 数据集与模型定义 =================
class SpectrogramDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        self.labels = [int(f.split('-')[1].split('.')[0]) for f in self.image_files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        return image, label


class CNN(nn.Module):
    def __init__(self, num_classes=6, base_channels=32, dropout_rate=0.3,
                 extra_conv_layers=1, use_res=True, kernel_size=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2)
        )
        current_channels = base_channels
        for _ in range(extra_conv_layers):
            self.features.append(nn.Conv2d(current_channels, current_channels * 2, kernel_size, padding=1))
            self.features.append(nn.BatchNorm2d(current_channels * 2))
            self.features.append(nn.LeakyReLU(0.1))
            self.features.append(nn.MaxPool2d(2, 2))
            current_channels *= 2

        if use_res:
            self.res = nn.Sequential(
                nn.Conv2d(current_channels, current_channels // 8, 1),
                nn.ReLU(),
                nn.Conv2d(current_channels // 8, current_channels, 1),
                nn.Sigmoid()
            )
        else:
            self.res = None

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(current_channels, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        if self.res is not None:
            res_mask = self.res(x)
            x = x * res_mask
        return self.classifier(x)


# ================= 3. 数据预处理与反归一化定义 =================
normalize_mean = [0.485, 0.456, 0.406]
normalize_std = [0.229, 0.224, 0.225]

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(normalize_mean, normalize_std)
])

inv_transform = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.], std=[1 / s for s in normalize_std]),
    transforms.Normalize(mean=[-m for m in normalize_mean], std=[1., 1., 1.])
])

# ================= 4. 初始化加载 =================
test_dir = '小波同步压缩变换时频/test_img'
test_dataset = SpectrogramDataset(test_dir, test_transform)

model = CNN(num_classes=6, base_channels=94, extra_conv_layers=0,
            use_res=False, dropout_rate=0.13046, kernel_size=5).to(device)
model.load_state_dict(torch.load('best_model_wsst.pth', map_location=device))
model.eval()

# ================= 5. 构建背景分布 =================
print("正在抽取测试集背景分布供 GradientExplainer 使用...")
background_tensors = []
for i in random.sample(range(len(test_dataset)), 200):
    img, _ = test_dataset[i]
    background_tensors.append(img)
background = torch.stack(background_tensors).to(device)
explainer = shap.GradientExplainer(model, background)

# ================= 6. 获取指定ID的样本并提取 SHAP 值与概率 =================
# 你指定要拼接的样本 ID (分别对应类别 1 到 6)
target_ids = [123, 329, 29, 124, 192, 269]

print("正在计算选定样本的 SHAP 值与预测概率...")

batch_pixels = []
batch_shaps_all = [[] for _ in range(6)]
batch_labels_7col = []  # 存储 7列图 (原图+6热力图) 的所有标签
batch_labels_2col = []  # 存储 2列图 (原图+1热力图) 的所有标签

for i, s_idx in enumerate(target_ids):
    true_label = i  # 0-5 对应 Type 1-6
    img_tensor, _ = test_dataset[s_idx]
    input_tensor = img_tensor.unsqueeze(0).to(device)

    # 获取 SHAP 值 (注意千万不能加 torch.no_grad)
    shap_values_ranked, indexes = explainer.shap_values(input_tensor, ranked_outputs=6)

    # [新增] 获取模型的预测概率
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    # 智能追踪类别维度
    shap_values_abs = [None] * 6
    for rank_pos in range(6):
        class_idx = indexes[0][rank_pos]
        if isinstance(shap_values_ranked, list):
            sv = shap_values_ranked[rank_pos]
        else:
            shape = shap_values_ranked.shape
            if shape[0] == 6:
                sv = shap_values_ranked[rank_pos]
            elif len(shape) > 1 and shape[1] == 6:
                sv = shap_values_ranked[:, rank_pos, ...]
            elif shape[-1] == 6:
                sv = shap_values_ranked[..., rank_pos]
            else:
                sv = shap_values_ranked[rank_pos]

        if sv.ndim == 3: sv = np.expand_dims(sv, axis=0)
        shap_values_abs[class_idx] = sv

    # 反归一化原图
    with torch.no_grad():
        orig_tensor = inv_transform(img_tensor.clone())
    orig_img = orig_tensor.permute(1, 2, 0).cpu().numpy()
    orig_img = np.clip(orig_img, 0, 1)
    batch_pixels.append(orig_img)

    # [核心] 排序：将真实标签(分类正确)强制放在第一列，其余的按顺序向后排
    ordered_indices = [true_label] + [c for c in range(6) if c != true_label]

    # 原图上方留空 (匹配你的参考图样式)
    row_7col = [""]
    row_2col = [""]

    for col_idx, class_idx in enumerate(ordered_indices):
        sv_transposed = np.transpose(shap_values_abs[class_idx], (0, 2, 3, 1))
        batch_shaps_all[col_idx].append(sv_transposed[0])

        # [新增] 格式化标签：类别名称 + 概率百分比
        prob_str = f"{probs[class_idx] * 100:.1f}%"
        label_text = f"{class_names[class_idx]}\n({prob_str})"
        row_7col.append(label_text)

        if col_idx == 0:  # 这是正确类别的 SHAP
            row_2col.append(label_text)

    batch_labels_7col.append(row_7col)
    batch_labels_2col.append(row_2col)

# 转换为 SHAP 可接受的 Numpy Arrays
batch_pixels = np.array(batch_pixels)
for col_idx in range(6):
    batch_shaps_all[col_idx] = np.array(batch_shaps_all[col_idx])

# ================= 全局字体与排版设置 =================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# ================= 7. 绘制多类别拼接图 (7列大图优化) =================
print("正在生成 '多类别拼接图' (上方居中带概率标签)...")
# 稍微减小 width，让整体不会因为过宽而导致各列被拉伸得太开
shap.image_plot(batch_shaps_all,
                pixel_values=batch_pixels,
                labels=None,
                width=20,
                show=False)

fig = plt.gcf()
axes = fig.axes
# 提取前 42 个子图 (6行 * 7列) 给它们加标题
for idx in range(42):
    ax = axes[idx]
    row = idx // 7
    col = idx % 7
    text = batch_labels_7col[row][col]
    # pad=4 缩小标题与图片的距离，使其紧凑
    ax.set_title(text, fontsize=14, fontname='Times New Roman', pad=4, loc='center')

# 先统一调整图片矩阵的布局，大幅度压缩 hspace (行距) 和 wspace (列距)
fig.subplots_adjust(wspace=0.02, hspace=0.25, left=0.02, right=0.98, top=0.92, bottom=0.15)

# [核心修复] 单独提取 SHAP 生成的最后一个坐标轴 (也就是底部的 Colorbar)
cb_ax = axes[-1]
pos = cb_ax.get_position()
# 强制将 Colorbar 往下平移 0.08 的绝对坐标距离，彻底告别穿模！
cb_ax.set_position([pos.x0, pos.y0 - 0.08, pos.width, pos.height * 0.8])

plt.savefig("combined_all_classes.svg", dpi=300, bbox_inches='tight', pad_inches=0.15)
plt.close()

# ================= 8. 绘制仅限真实类别的拼接图 (自定义 3x4 布局) =================
print("正在生成 '真实类别拼接图' (3x4 布局，无 Colorbar)...")

# 提取 SHAP 的经典红蓝透明渐变色 (如果没有，退化为 RdBu_r)
try:
    from shap.plots.colors import red_transparent_blue

    cmap = red_transparent_blue
except ImportError:
    cmap = 'RdBu_r'

# 纯手工构建 3行 4列 的完美画板
fig2, axes2 = plt.subplots(3, 4, figsize=(12, 10))

for i in range(6):
    # 计算当前样本应该放在哪一行、那一列
    row = i // 2
    col_base = (i % 2) * 2  # 左边原图的列索引 (0 或 2)

    ax_orig = axes2[row, col_base]
    ax_shap = axes2[row, col_base + 1]

    # --- 1. 绘制原图 ---
    img = batch_pixels[i]
    ax_orig.imshow(img)
    ax_orig.axis('off')
    # 对于原图，只标明 Type X
    ax_orig.set_title(f"Type {i + 1}", fontsize=15, fontname='Times New Roman', pad=6)

    # --- 2. 绘制热力图 ---
    shap_val = batch_shaps_all[0][i]  # shape: (224, 224, 3)

    # 模拟 SHAP 官方叠加逻辑：原图灰度化作为底图
    gray_img = img.mean(axis=-1)
    ax_shap.imshow(gray_img, cmap='gray', alpha=0.15)

    # SHAP 值三通道叠加
    shap_sum = shap_val.sum(axis=-1)
    max_val = np.max(np.abs(shap_sum)) + 1e-8  # 提取绝对值的最大值作为色标极值

    # 将热力图覆盖在灰度图上 (透明色带使得灰色底纹能透出来)
    ax_shap.imshow(shap_sum, cmap=cmap, vmin=-max_val, vmax=max_val)
    ax_shap.axis('off')

    # 热力图的标题采用之前生成好的带概率的文本 (如: Type 1 \n (99.9%))
    text = batch_labels_2col[i][1]
    ax_shap.set_title(text, fontsize=15, fontname='Times New Roman', pad=6)

# 手动调整 3x4 布局的留白和间距
fig2.subplots_adjust(wspace=0.05, hspace=0.30, left=0.02, right=0.98, top=0.95, bottom=0.05)
plt.savefig("combined_true_class_only.tif", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.close()

print("🎉 两张拼接热力图已成功保存！大图不再穿透且已压紧，小图已完美转为 3x4 无横条布局！")