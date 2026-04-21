import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import optuna
from optuna.samplers import NSGAIISampler
from optuna.visualization import plot_pareto_front, plot_parallel_coordinate
from optuna.trial import TrialState
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image


# 设置随机种子
def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),

])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


class SpectrogramDataset:
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



class OptimizedCNN(nn.Module):
    def __init__(self, num_classes=6, base_channels=32, dropout_rate=0.3,
                 extra_conv_layers=1, use_attention=True, kernel_size=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2)
        )

        # 动态添加卷积层
        current_channels = base_channels
        for _ in range(extra_conv_layers):
            self.features.append(nn.Conv2d(current_channels, current_channels * 2, kernel_size, padding=1))
            self.features.append(nn.BatchNorm2d(current_channels * 2))
            self.features.append(nn.LeakyReLU(0.1))
            self.features.append(nn.MaxPool2d(2, 2))
            current_channels *= 2

        if use_attention:
            self.attention = nn.Sequential(
                nn.Conv2d(current_channels, current_channels // 8, 1),
                nn.ReLU(),
                nn.Conv2d(current_channels // 8, current_channels, 1),
                nn.Sigmoid()
            )
        else:
            self.attention = None

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
        if self.attention is not None:
            attention_mask = self.attention(x)
            x = x * attention_mask
        return self.classifier(x)

    def get_features(self, x):
        x = self.features(x)
        if self.attention is not None:
            attention_mask = self.attention(x)
            x = x * attention_mask
        return torch.flatten(nn.AdaptiveAvgPool2d(1)(x), start_dim=1)

# Optuna目标函数
def objective(trial):
    params = {
        'base_channels': trial.suggest_int('base_channels', 32, 128),
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'dropout_rate': trial.suggest_float('dropout', 0.1, 0.5),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'kernel_size': trial.suggest_int('kernel_size', 3, 7, step=2),
        'extra_conv_layers': trial.suggest_int('extra_conv_layers', 0, 2),
        'use_attention': trial.suggest_categorical('use_attention', [True, False]),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64])
    }

    model = OptimizedCNN(
        num_classes=6,
        base_channels=params['base_channels'],
        kernel_size=params['kernel_size'],
        dropout_rate=params['dropout_rate'],
        extra_conv_layers=params['extra_conv_layers'],
        use_attention=params['use_attention']
    ).to(device)

    params_count = sum(p.numel() for p in model.parameters())

    # 数据加载
    train_loader = DataLoader(
        SpectrogramDataset('小波同步压缩变换时频/train_img', train_transform),
        batch_size=params['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        SpectrogramDataset('小波同步压缩变换时频/valid_img', test_transform),
        batch_size=params['batch_size'],
        shuffle=False
    )

    # 优化器配置
    optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    patience = 5
    no_improve = 0

    # 训练循环
    epoches = 60
    for epoch in range(epoches):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted.cpu() == labels.cpu()).sum().item()

        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best_val_acc, params_count


# 增强的可视化函数
def visualize_study(study):
    trials = [t for t in study.trials if t.state == TrialState.COMPLETE and t.values is not None]

    print(f"总试验数: {len(study.trials)}")
    print(f"有效试验数: {len(trials)}")

    # 帕累托前沿图
    fig = plot_pareto_front(
        study,
        targets=lambda t: (t.values[0], t.values[1]),
        target_names=["Validation Accuracy", "Parameter Count"],
        include_dominated_trials=False
    )
    fig.update_layout(width=800, height=600)
    fig.write_image("pareto_front.png", engine="kaleido")

    # 平行坐标图
    fig = plot_parallel_coordinate(
        study,
        params=["base_channels", "extra_conv_layers", "lr", "dropout", "use_attention"],
        target=lambda t: t.values[0]
    )
    fig.update_layout(width=1200, height=800)
    fig.write_image("parallel_coordinates.png", engine="kaleido")

    # 3D优化景观图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    x = [t.params['base_channels'] for t in trials]
    y = [t.values[0] for t in trials]
    z = [t.values[1] for t in trials]
    sc = ax.scatter(x, y, z, c=z, cmap='viridis', marker='o')
    ax.set_xlabel('Base Channels')
    ax.set_ylabel('Accuracy')
    ax.set_zlabel('Parameters')
    plt.colorbar(sc, label='Parameter Count')
    plt.savefig("3d_optimization.png")
    plt.close()


# 最终评估函数
def final_evaluation(model, test_loader):
    print("正在测试最终模型")
    model.eval()
    all_features = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            features = model.get_features(inputs)
            outputs = model(inputs)

            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    # t-SNE可视化
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    features_2d = tsne.fit_transform(np.concatenate(all_features))

    plt.figure(figsize=(10, 8))
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=all_labels, cmap='tab10', alpha=0.6)
    plt.colorbar(ticks=range(6))
    plt.title('t-SNE Visualization of Feature Embeddings')
    plt.savefig('tsne_visualization.png')
    plt.close()

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()

    return all_labels, all_preds


# 主流程
if __name__ == "__main__":
    study = optuna.create_study(
        directions=['maximize', 'minimize'],
        sampler=NSGAIISampler(population_size=30)
    )
    study.optimize(objective, n_trials=40, show_progress_bar=True)

    visualize_study(study)

    # 选择最佳试验
    best_trial = max(study.best_trials, key=lambda t: t.values[0] - t.values[1] / 1e6)

    # 保存最佳超参数到txt文件
    with open('best_hyperparameters.txt', 'w') as f:
        f.write(f"=== Best Hyperparameters ===\n")
        f.write(f"Validation Accuracy: {best_trial.values[0]:.4f}\n")
        f.write(f"Parameter Count: {int(best_trial.values[1])}\n\n")
        f.write("Hyperparameters:\n")
        for key, value in best_trial.params.items():
            # 对浮点参数保留更多小数位
            if isinstance(value, float):
                f.write(f"{key}: {value:.6f}\n")
            else:
                f.write(f"{key}: {value}\n")
        f.write("\n=== Parameter Description ===\n")
        f.write("base_channels: 初始卷积通道数\n")
        f.write("lr: 学习率\n")
        f.write("dropout: Dropout概率\n")
        f.write("weight_decay: 权重衰减系数\n")
        f.write("kernel_size: 卷积核尺寸\n")
        f.write("extra_conv_layers: 额外卷积层数\n")
        f.write("use_attention: 是否使用注意力机制\n")
        f.write("batch_size: 批量大小\n")

    # 训练最终模型
    print("最佳参数,base_channel",best_trial.params['base_channels'])
    print("dropout", best_trial.params['dropout'])
    print("lr", best_trial.params['lr'])
    print("kernel_size", best_trial.params['kernel_size'])
    print("extra_conv_layers", best_trial.params['extra_conv_layers'])
    print("use_attention", best_trial.params['use_attention'])
    print("batch_size", best_trial.params['batch_size'])
    print("weight_decay", best_trial.params['weight_decay'])


    final_model = OptimizedCNN(
        num_classes=6,
        base_channels=best_trial.params['base_channels'],
        dropout_rate=best_trial.params['dropout'],
        kernel_size=best_trial.params['kernel_size'],
        extra_conv_layers=best_trial.params['extra_conv_layers'],
        use_attention=best_trial.params['use_attention']
    ).to(device)

    # 完整训练数据
    full_dataset = torch.utils.data.ConcatDataset([
        SpectrogramDataset('小波同步压缩变换时频/train_img', train_transform),
        SpectrogramDataset('小波同步压缩变换时频/valid_img', test_transform)
    ])
    train_loader = DataLoader(full_dataset, batch_size=best_trial.params['batch_size'], shuffle=True)

    # 优化器配置
    optimizer = optim.AdamW(final_model.parameters(),
                            lr=best_trial.params['lr'],
                            weight_decay=best_trial.params['weight_decay'])

    # 完整训练循环
    print("正在训练最终模型")
    epoches = 60
    for epoch in range(epoches):
        final_model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = final_model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()

    # 最终评估
    test_loader = DataLoader(
        SpectrogramDataset('小波同步压缩变换时频/test_img', test_transform),
        batch_size=best_trial.params['batch_size'],
        shuffle=False
    )
    true_labels, pred_labels = final_evaluation(final_model, test_loader)

    # 打印评估指标
    print(f"Test Accuracy: {np.mean(np.array(true_labels) == np.array(pred_labels)):.4f}")
    print("Confusion Matrix and t-SNE visualization saved.")