import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from PIL import Image
from sklearn.metrics import f1_score, recall_score, precision_score
from tqdm import tqdm


def seed_everything(seed=42):
    """设置随机种子以确保实验可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 数据增强和预处理 (无变动) ---
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=6),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# --- 数据集类 (无变动) ---
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
        return self.transform(image) if self.transform else image, label


# --- CNN 模型定义 ---
class CNN(nn.Module):
    def __init__(self, num_classes=6, base_channels=32, dropout_rate=0.3,
                 extra_conv_layers=1, use_res=True, kernel_size=3):
        super().__init__()
        self.features = self._make_layers(base_channels, extra_conv_layers, kernel_size)

        if use_res:
            self.attention = nn.Sequential(
                nn.Conv2d(base_channels * (2 ** extra_conv_layers),
                          base_channels * (2 ** extra_conv_layers) // 8, 1),
                nn.ReLU(),
                nn.Conv2d(base_channels * (2 ** extra_conv_layers) // 8,
                          base_channels * (2 ** extra_conv_layers), 1),
                nn.Sigmoid()
            )
        else:
            self.attention = None

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * (2 ** extra_conv_layers), 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def _make_layers(self, base_channels, num_layers, kernel_size):
        layers = []
        current_channels = base_channels
        layers += [
            nn.Conv2d(3, current_channels, kernel_size, padding=1),
            nn.BatchNorm2d(current_channels),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2)
        ]

        for _ in range(num_layers):
            layers += [
                nn.Conv2d(current_channels, current_channels * 2, kernel_size, padding=1),
                nn.BatchNorm2d(current_channels * 2),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(2, 2)
            ]
            current_channels *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        if self.attention is not None:
            x = x * self.attention(x)
        return self.classifier(x)


# --- 实验运行函数 (无变动) ---
def run_experiment(seed, num_epochs=60, visualize=False):
    seed_everything(seed)

    # 数据加载
    train_dataset = SpectrogramDataset('小波同步压缩变换时频/train_img',
                                       train_transform)
    val_dataset = SpectrogramDataset('小波同步压缩变换时频/valid_img', test_transform)
    test_dataset = SpectrogramDataset('小波同步压缩变换时频/test_img', test_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 模型初始化
    model = CNN(num_classes=6, base_channels=94, extra_conv_layers=0,
                use_res=False, dropout_rate=0.1370956, kernel_size=5).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.000357, weight_decay=0.000931)

    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    best_val_acc = 0.0
    best_model_state = None
    patience = 18
    no_improve = 0

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_bar = tqdm(train_loader, desc=f'Train Epoch {epoch + 1}/{num_epochs}',
                         bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_size = inputs.size(0)
            train_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += batch_size
            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{(predicted == labels).sum().item() / batch_size:.3f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
        train_loss /= train_total
        train_acc = train_correct / train_total

        # 验证步骤
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_bar = tqdm(val_loader, desc='Validating', bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', leave=False)
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                batch_size = inputs.size(0)
                val_loss += loss.item() * batch_size
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += batch_size
                val_bar.set_postfix({
                    'val_loss': f'{loss.item():.4f}',
                    'val_acc': f'{(predicted == labels).sum().item() / batch_size:.3f}'
                })
        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"\nEpoch {epoch + 1:02d}/{num_epochs} Summary:")
        print(
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # 早停机制
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            no_improve = 0
            print(f"🔥 New best validation accuracy: {val_acc:.4f}")
        else:
            no_improve += 1
            print(f"⏳ No improvement for {no_improve}/{patience} epochs")
            if no_improve >= patience:
                print(f"🛑 Early stopping at epoch {epoch + 1}")
                break
        scheduler.step()

    # 加载最佳模型进行测试
    model.load_state_dict(best_model_state)
    test_acc, test_f1, test_recall, test_precision = evaluate_model(model, test_loader, seed, visualize)
    return test_acc, test_f1, test_recall, test_precision


def evaluate_model(model, test_loader, seed, visualize):
    """
    模型评估函数，增加了保存t-SNE和混淆矩阵数据的逻辑
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_features = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            # 提取特征
            features_output = model.features(inputs)
            pooled_features = nn.AdaptiveAvgPool2d(1)(features_output)
            flattened_features = torch.flatten(pooled_features, start_dim=1)
            # 获取模型最终预测
            outputs = model.classifier(pooled_features)

            all_features.append(flattened_features.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    # 计算所有指标
    accuracy = np.mean(np.array(all_labels) == np.array(all_preds))
    f1 = f1_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')

    if visualize:
        # 准备数据
        all_features_np = np.concatenate(all_features)
        all_labels_np = np.array(all_labels)
        all_preds_np = np.array(all_preds)

        # ------------------- t-SNE 可视化与数据保存 -------------------
        print(f"\n--- [Seed {seed}] Generating and saving t-SNE ---")
        # 1. 保存t-SNE的原始数据
        tsne_data_path = f'tsne_data_seed_{seed}.npz'
        np.savez(tsne_data_path,
                 features=all_features_np,
                 labels=all_labels_np,
                 preds=all_preds_np)
        print(f"✅ Saved t-SNE data to: {tsne_data_path}")

        # 2. 计算并绘制t-SNE图像
        tsne = TSNE(n_components=2, perplexity=30, random_state=seed)
        features_2d = tsne.fit_transform(all_features_np)

        plt.figure(figsize=(10, 8))
        unique_labels = np.unique(all_labels_np)
        cmap = plt.get_cmap('tab10', len(unique_labels))
        for label in unique_labels:
            mask = (all_labels_np == label)
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], color=cmap(label),
                        label=f'Class {label}', alpha=0.6, edgecolors='w')
        plt.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f't-SNE Visualization (Seed {seed})')
        plt.tight_layout()
        tsne_img_path = f'tsne_seed_{seed}.png'
        plt.savefig(tsne_img_path, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved t-SNE plot to: {tsne_img_path}")

        # ------------------- 混淆矩阵可视化与数据保存 -------------------
        print(f"--- [Seed {seed}] Generating and saving Confusion Matrix ---")
        # 1. 计算混淆矩阵并保存.npy文件
        cm = confusion_matrix(all_labels_np, all_preds_np)
        cm_data_path = f'confusion_matrix_seed_{seed}.npy'
        np.save(cm_data_path, cm)
        print(f"✅ Saved confusion matrix data to: {cm_data_path}")

        # 2. 绘制混淆矩阵图像
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title(f'Confusion Matrix (Seed {seed})')
        cm_img_path = f'cm_seed_{seed}.png'
        plt.savefig(cm_img_path)
        plt.close()
        print(f"✅ Saved confusion matrix plot to: {cm_img_path}")

    return accuracy, f1, recall, precision


if __name__ == "__main__":
    # ========== 修改点 1: 设置运行次数为 10 ==========
    num_runs = 10
    seeds = [42 + i for i in range(num_runs)]

    accuracies, f1_scores, recalls, precisions = [], [], [], []

    # 创建并打开结果文件
    with open('experiment_results.txt', 'w') as f:
        f.write("========== 实验报告 ==========\n\n")
        f.write(f"运行次数: {num_runs}\n")
        f.write(f"使用的随机种子: {seeds}\n\n")

        for i, seed in enumerate(seeds):
            print(f"\n{'=' * 20} Running with seed {seed} ({i + 1}/{num_runs}) {'=' * 20}")
            f.write(f"\n=== 第 {i + 1}/{num_runs} 次运行（种子 {seed}） ===\n")

            # ========== 修改点 2: 总是进行可视化和保存 ==========
            acc, f1, recall, precision = run_experiment(seed, num_epochs=60, visualize=True)

            accuracies.append(acc)
            f1_scores.append(f1)
            recalls.append(recall)
            precisions.append(precision)

            # 打印并保存详细结果
            print(f"\n--- [Seed {seed}] Final Results ---")
            print(f"Test Accuracy: {acc:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"Precision: {precision:.4f}")

            f.write(f"测试准确率: {acc:.4f}\n")
            f.write(f"F1分数: {f1:.4f}\n")
            f.write(f"召回率: {recall:.4f}\n")
            f.write(f"精确率: {precision:.4f}\n")

        # 计算并写入最终统计信息
        final_report = [
            "\n========== 最终统计结果 ==========",
            f"平均测试准确率: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}",
            f"平均F1分数: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}",
            f"平均召回率: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}",
            f"平均精确率: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}",
            "\n详细数据:",
            "准确率列表: " + ", ".join([f"{x:.4f}" for x in accuracies]),
            "F1分数列表: " + ", ".join([f"{x:.4f}" for x in f1_scores]),
            "召回率列表: " + ", ".join([f"{x:.4f}" for x in recalls]),
            "精确率列表: " + ", ".join([f"{x:.4f}" for x in precisions])
        ]

        report_str = "\n".join(final_report)
        print(report_str)
        f.write(report_str)

    print("\n🚀 实验完成！结果已保存到 experiment_results.txt，所有图片和数据文件已生成在当前目录。")