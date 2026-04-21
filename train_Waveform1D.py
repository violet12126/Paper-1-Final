import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import random
from tqdm import tqdm

# ========== 固定的最优超参数 ==========
BEST_PARAMS = {
    'base_channels': 44,
    'lr': 0.00015257679533429273,
    'dropout': 0.19322850860193252,
    'weight_decay': 1.1481369329951204e-06,
    'kernel_size': 3,
    'extra_conv_layers': 0,
    'use_attention': True,
    'batch_size': 32
}


def seed_everything(seed=42):
    """设置随机种子以确保实验可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 数据加载机制 ---
def load_excel_data(file_path):
    print("正在加载 Excel 数据，这可能需要一点时间...")
    all_sheets = pd.read_excel(file_path, sheet_name=None, header=None)

    X_list = []
    y_raw_list = []

    for sheet_name, df in all_sheets.items():
        # 假设第一列是标签，后面是特征
        labels = df.iloc[:, 0].values
        features = df.iloc[:, 1:].values.astype(np.float32)

        X_list.append(features)
        y_raw_list.append(labels)

    X = np.vstack(X_list)
    y_raw = np.concatenate(y_raw_list)

    # 使用 LabelEncoder 自动将任何格式的标签映射为 0 到 n-1
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    print(f"数据加载完成！总样本数: {X.shape[0]}, 序列长度: {X.shape[1]}")
    print(f"检测到的原始标签种类: {encoder.classes_}")
    print(f"映射后的内部训练标签: {np.unique(y)}")

    return X, y, len(encoder.classes_)


# --- 一维序列数据集类 ---
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        # 将数据扩展出 channel 维度 (Batch, 1, Seq_Length)
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# --- 1D CNN 模型定义 ---
class CNN1D(nn.Module):
    def __init__(self, num_classes, base_channels, dropout_rate,
                 extra_conv_layers, use_attention, kernel_size):
        super().__init__()

        self.features = self._make_layers(base_channels, extra_conv_layers, kernel_size)
        current_channels = base_channels * (2 ** extra_conv_layers)

        if use_attention:
            self.attention = nn.Sequential(
                nn.Conv1d(current_channels, max(1, current_channels // 8), 1),
                nn.ReLU(),
                nn.Conv1d(max(1, current_channels // 8), current_channels, 1),
                nn.Sigmoid()
            )
        else:
            self.attention = None

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(current_channels, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def _make_layers(self, base_channels, num_layers, kernel_size):
        layers = []
        current_channels = base_channels
        padding = kernel_size // 2

        layers += [
            nn.Conv1d(1, current_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(current_channels),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2, 2)
        ]

        for _ in range(num_layers):
            layers += [
                nn.Conv1d(current_channels, current_channels * 2, kernel_size, padding=padding),
                nn.BatchNorm1d(current_channels * 2),
                nn.LeakyReLU(0.1),
                nn.MaxPool1d(2, 2)
            ]
            current_channels *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        if self.attention is not None:
            x = x * self.attention(x)
        return self.classifier(x)


# --- 实验运行函数 ---
def run_experiment(X_all, y_all, num_classes, seed, num_epochs=60):
    seed_everything(seed)

    # 在函数内部划分数据以确保每个 seed 的划分可以不同（或者相同，由于传入seed），确保交叉验证的多样性
    X_temp, X_test, y_temp, y_test = train_test_split(X_all, y_all, test_size=0.15, random_state=seed, stratify=y_all)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15 / 0.85, random_state=seed,
                                                      stratify=y_temp)

    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)
    test_dataset = SequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BEST_PARAMS['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BEST_PARAMS['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BEST_PARAMS['batch_size'], shuffle=False)

    # 模型初始化加载最新参数
    model = CNN1D(
        num_classes=num_classes,
        base_channels=BEST_PARAMS['base_channels'],
        dropout_rate=BEST_PARAMS['dropout'],
        extra_conv_layers=BEST_PARAMS['extra_conv_layers'],
        use_attention=BEST_PARAMS['use_attention'],
        kernel_size=BEST_PARAMS['kernel_size']
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=BEST_PARAMS['lr'], weight_decay=BEST_PARAMS['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    best_val_acc = 0.0
    best_model_state = None
    patience = 18
    no_improve = 0

    # ========== 新增：初始化用于保存训练曲线的列表 ==========
    history_metrics = []

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

        print(
            f"\nEpoch {epoch + 1:02d}/{num_epochs} Summary: Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # ========== 新增：将当前 epoch 的指标保存到列表中 ==========
        history_metrics.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc
        })

        # 早停机制
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            no_improve = 0
            print(f"🔥 New best validation accuracy: {val_acc:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"🛑 Early stopping at epoch {epoch + 1}")
                break
        scheduler.step()

    # ========== 新增：训练循环结束后，将记录的数据保存为 CSV ==========
    csv_filename = f'training_curve_seed_{seed}.csv'
    pd.DataFrame(history_metrics).to_csv(csv_filename, index=False)
    print(f"📊 Training curve saved to {csv_filename}")

    # 加载最佳模型进行测试
    model.load_state_dict(best_model_state)
    test_acc, test_f1, test_recall, test_precision = evaluate_model(model, test_loader, num_classes, seed)
    return test_acc, test_f1, test_recall, test_precision


# --- 评估与可视化函数 ---
def evaluate_model(model, test_loader, num_classes, seed):
    model.eval()
    all_labels = []
    all_preds = []
    all_features = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)

            # 提取中间特征并降维 (使用AdaptiveAvgPool1d)
            features_output = model.features(inputs)
            if model.attention is not None:
                features_output = features_output * model.attention(features_output)

            pooled_features = nn.AdaptiveAvgPool1d(1)(features_output)
            flattened_features = torch.flatten(pooled_features, start_dim=1)

            # 最终预测
            outputs = model.classifier[1:](pooled_features)  # 跳过内部池化，直接接后续Linear层预测

            all_features.append(flattened_features.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    # 计算所有指标
    accuracy = np.mean(np.array(all_labels) == np.array(all_preds))
    f1 = f1_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')

    # 准备数据
    all_features_np = np.concatenate(all_features)
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)

    # ------------------- t-SNE 可视化与数据保存 -------------------
    print(f"\n--- [Seed {seed}] Generating and saving t-SNE ---")
    tsne_data_path = f'tsne_data_seed_{seed}.npz'
    np.savez(tsne_data_path, features=all_features_np, labels=all_labels_np, preds=all_preds_np)

    # 计算并绘制t-SNE图像
    tsne = TSNE(n_components=2, perplexity=min(30, len(all_labels_np) - 1), random_state=seed)
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
    plt.savefig(f'tsne_seed_{seed}.png', bbox_inches='tight')
    plt.close()

    # ------------------- 混淆矩阵可视化与数据保存 -------------------
    print(f"--- [Seed {seed}] Generating and saving Confusion Matrix ---")
    cm = confusion_matrix(all_labels_np, all_preds_np)
    np.save(f'confusion_matrix_seed_{seed}.npy', cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix (Seed {seed})')
    plt.savefig(f'cm_seed_{seed}.png')
    plt.close()

    return accuracy, f1, recall, precision


if __name__ == "__main__":
    # ========== 修改点 1: 在主循环外加载一次数据 ==========
    # 请务必替换为你的实际 Excel 路径
    DATA_PATH = 'output.xlsx'
    X_all, y_all, num_classes = load_excel_data(DATA_PATH)

    num_runs = 1
    seeds = [42 + i for i in range(num_runs)]
    accuracies, f1_scores, recalls, precisions = [], [], [], []

    # 创建并打开结果文件
    with open('experiment_results.txt', 'w', encoding='utf-8') as f:
        f.write("========== 实验报告 ==========\n\n")
        f.write("使用的超参数:\n")
        for k, v in BEST_PARAMS.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\n运行次数: {num_runs}\n")
        f.write(f"使用的随机种子: {seeds}\n\n")

        for i, seed in enumerate(seeds):
            print(f"\n{'=' * 20} Running with seed {seed} ({i + 1}/{num_runs}) {'=' * 20}")
            f.write(f"\n=== 第 {i + 1}/{num_runs} 次运行（种子 {seed}） ===\n")

            # ========== 运行实验 ==========
            acc, f1, recall, precision = run_experiment(X_all, y_all, num_classes, seed, num_epochs=60)

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

    print(
        "\n🚀 实验完成！结果已保存到 experiment_results.txt，所有的图片、数据文件以及各个 Seed 的训练曲线 CSV 已生成在当前目录。")
