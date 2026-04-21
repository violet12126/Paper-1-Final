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

# ================= 5. 构建背景分布 (GradientExplainer 专属) =================
print("正在抽取测试集背景分布供 GradientExplainer 使用...")
background_tensors = []
# 随机抽取 200 张图作为求梯度的基准线
for i in random.sample(range(len(test_dataset)), 200):
    img, _ = test_dataset[i]
    background_tensors.append(img)
background = torch.stack(background_tensors).to(device)

# 初始化梯度解释器 (直接传入模型和标准化后的背景张量)
explainer = shap.GradientExplainer(model, background)


# ================= 6. 扫描所有正确样本 =================
def get_all_correct_samples(dataset, model):
    correct_pools = {i: [] for i in range(6)}
    print("正在扫描测试集，收集所有分类正确的样本...")

    with torch.no_grad():
        for idx in range(len(dataset)):
            img, label = dataset[idx]
            output = model(img.unsqueeze(0).to(device))
            pred = torch.argmax(output, dim=1).item()
            if pred == label:
                correct_pools[label].append(idx)

    print("\n--- 样本筛选统计 ---")
    for label in range(6):
        print(f"Type {label + 1}: 共找到 {len(correct_pools[label])} 张正确样本")
    print("--------------------\n")
    return correct_pools


# ================= 7. SHAP 核心绘制函数 =================
def explain_and_plot(img_tensor, true_label, sample_idx, output_path_prefix):
    input_tensor = img_tensor.unsqueeze(0).to(device)

    # 1. 提取真实概率
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    # 2. 直接获取 SHAP 梯度值
    shap_values_ranked, indexes = explainer.shap_values(input_tensor, ranked_outputs=6)

    # -------- 【智能追踪类别维度】 --------
    shap_values_abs = [None] * 6
    for rank_pos in range(6):
        class_idx = indexes[0][rank_pos]

        if isinstance(shap_values_ranked, list):
            sv = shap_values_ranked[rank_pos]
        elif isinstance(shap_values_ranked, np.ndarray):
            shape = shap_values_ranked.shape
            if shape[0] == 6:
                sv = shap_values_ranked[rank_pos]
            elif len(shape) > 1 and shape[1] == 6:
                sv = shap_values_ranked[:, rank_pos, ...]
            elif shape[-1] == 6:
                sv = shap_values_ranked[..., rank_pos]
            else:
                sv = shap_values_ranked[rank_pos]

        # 确保 sv 的形状是 (1, 3, 224, 224) 供后续转置
        if sv.ndim == 3:
            sv = np.expand_dims(sv, axis=0)

        shap_values_abs[class_idx] = sv

    # 转换维度供绘图使用: 从 (1, 3, 224, 224) 变为 (1, 224, 224, 3)
    shap_values_transposed = [np.transpose(sv, (0, 2, 3, 1)) for sv in shap_values_abs]

    # 3. 反归一化原图用于背景显示
    with torch.no_grad():
        orig_tensor = inv_transform(img_tensor.clone())
    orig_img = orig_tensor.permute(1, 2, 0).cpu().numpy()
    orig_img = np.clip(orig_img, 0, 1)
    orig_img_plot = np.expand_dims(orig_img, axis=0)

    # 4. 强制将 true_label 放第一列
    ordered_indices = [true_label] + [i for i in range(6) if i != true_label]

    shap_values_ordered = [shap_values_transposed[i] for i in ordered_indices]
    ordered_class_names = [class_names[i] for i in ordered_indices]
    ordered_class_names_with_probs = [
        f"{class_names[i]}\n({probs[i] * 100:.1f}%)" for i in ordered_indices
    ]

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'

    # --- 绘制并保存 ---
    shap.image_plot(shap_values_ordered,
                    pixel_values=orig_img_plot,
                    labels=np.array([ordered_class_names]),
                    show=False)
    plt.savefig(f"{output_path_prefix}_idx{sample_idx}_original.png", dpi=300, bbox_inches='tight')
    plt.close()

    shap.image_plot(shap_values_ordered,
                    pixel_values=orig_img_plot,
                    labels=np.array([ordered_class_names_with_probs]),
                    show=False)
    plt.savefig(f"{output_path_prefix}_idx{sample_idx}_with_probs.png", dpi=300, bbox_inches='tight')
    plt.close()


# ================= 主执行流程 (按文件夹保存全部) =================
# 主输出根目录
output_root_dir = "shap_results_all"
os.makedirs(output_root_dir, exist_ok=True)

# 扫描获取所有正确的样本索引字典
correct_pools = get_all_correct_samples(test_dataset, model)

# 遍历每一个类别
for label in range(6):
    indices = correct_pools[label]
    if len(indices) == 0:
        continue

    # 为当前类别创建独立的子文件夹，如 "shap_results_all/Type_1"
    class_dir = os.path.join(output_root_dir, f'Type_{label + 1}')
    os.makedirs(class_dir, exist_ok=True)

    print(f"============== 开始绘制 {class_names[label]} 的热力图 (共 {len(indices)} 张) ==============")

    # 遍历该类别下的所有正确样本并绘图
    for i, s_idx in enumerate(indices):
        img_tensor, _ = test_dataset[s_idx]
        # 文件命名为：该文件夹下的 idx_XXX_original.png 等
        output_prefix = os.path.join(class_dir, f'idx_{s_idx}')

        explain_and_plot(img_tensor, label, s_idx, output_prefix)

        # 打印进度条
        if (i + 1) % 5 == 0 or (i + 1) == len(indices):
            print(f"  [{class_names[label]}] 进度: {i + 1} / {len(indices)} 张已完成")

    print(f"🎯 {class_names[label]} 的所有热力图已成功保存至: {class_dir}\n")

print("🎉 所有类别的正确样本 SHAP 热力图全部绘制完毕！")

