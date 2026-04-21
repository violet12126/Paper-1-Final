import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import csv

# 设置随机种子
def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed= 42
seed_everything(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 数据增强和预处理
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# 自定义数据集类
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
                 extra_conv_layers=1, use_res=True,kernel_size=3):
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

    def get_features(self, x):
        x = self.features(x)
        if self.res is not None:
            res_mask = self.res(x)
            x = x * res_mask
        return torch.flatten(nn.AdaptiveAvgPool2d(1)(x), start_dim=1)
class CNN2(nn.Module):
    def __init__(self, num_classes=6, base_channels=32, dropout_rate=0.3,
                 extra_conv_layers=1, use_res=True,kernel_size=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # 动态添加卷积层
        current_channels = base_channels
        for _ in range(extra_conv_layers):
            self.features.append(nn.Conv2d(current_channels, current_channels * 2, kernel_size, padding=1))
            self.features.append(nn.BatchNorm2d(current_channels * 2))
            self.features.append(nn.ReLU())
            self.features.append(nn.MaxPool2d(2, 2))
            current_channels *= 2

        if use_res:
            self.res = nn.Sequential(
                nn.Conv2d(current_channels, current_channels // 8, 1),
                nn.ReLU(),
                nn.Conv2d(current_channels // 8, current_channels, 1),
                nn.ReLU()
            )
        else:
            self.res = None

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # nn.Flatten(),   # ← 去掉这个
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
        x = nn.AdaptiveAvgPool2d(1)(x)
        x = torch.flatten(x, 1)  # ← 这里显式 flatten
        return self.classifier[1:](x)  # 跳过 classifier[0] 的 AdaptiveAvgPool2d

    def get_features(self, x):
        x = self.features(x)
        if self.res is not None:
            res_mask = self.res(x)
            x = x * res_mask
        return torch.flatten(nn.AdaptiveAvgPool2d(1)(x), start_dim=1)


# 数据加载
train_dataset = SpectrogramDataset('小波同步压缩变换时频/train_img', train_transform)
val_dataset = SpectrogramDataset('小波同步压缩变换时频/valid_img', test_transform)
test_dataset = SpectrogramDataset('小波同步压缩变换时频/test_img', test_transform)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 模型初始化
model = CNN(num_classes=6, base_channels=94, extra_conv_layers=0,
            use_res=False, dropout_rate=0.13046,kernel_size=5).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.00035737, weight_decay= 0.000931)

params_count = sum(p.numel() for p in model.parameters())
print(params_count)

criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

# 训练参数
num_epochs = 60
best_val_acc = 0.0
train_losses, val_losses = [], []
train_accs, val_accs = [], []

with open('wsst.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])

# 训练循环
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / total
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # 验证
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = val_loss / total
    val_acc = correct / total
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    # 更新学习率
    scheduler.step()

    # 添加早停机制
    patience = 15
    no_improve = 0
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        no_improve = 0
        torch.save(model.state_dict(), 'best_model_wsst.pth')
    else:
        no_improve += 1
        if no_improve >= patience:
            print("Early stopping!")
            break

    with open('wsst.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch + 1,
            train_loss,
            val_loss,
            train_acc,
            val_acc
        ])

    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    print('-' * 50)
# 绘制训练曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Validation')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train')
plt.plot(val_accs, label='Validation')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('training_curves.png')
plt.close()


# 测试和可视化
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

    # 保存t-SNE数据
    np.savez('tsne_data.npz',
             features=np.concatenate(all_features),
             labels=all_labels,
             preds=all_preds)

    # t-SNE可视化
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    features_2d = tsne.fit_transform(np.concatenate(all_features))

    # 设置全局字体
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'

    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(all_labels)
    cmap = plt.get_cmap('tab10', len(unique_labels))

    # 绘制每个类别的点
    for label in unique_labels:
        mask = np.array(all_labels) == label
        plt.scatter(features_2d[mask, 0],
                    features_2d[mask, 1],
                    color=cmap(label),
                    label=f'Type {label + 1}',  # 显示1-6
                    alpha=0.6,
                    edgecolors='w')

    # 设置坐标轴
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    # 确保显示坐标轴端点
    x_min, x_max = np.min(features_2d[:, 0]), np.max(features_2d[:, 0])
    y_min, y_max = np.min(features_2d[:, 1]), np.max(features_2d[:, 1])
    plt.xlim(x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min))
    plt.ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

    plt.title(f't-SNE Visualization (Seed {seed})', fontsize=14)
    plt.legend(title='Fault Types',
               bbox_to_anchor=(1.05, 1),
               loc='upper left',
               prop={'size': 10})
    plt.tight_layout()
    plt.savefig(f'tsne_seed_{seed}.png', bbox_inches='tight', dpi=300)
    plt.close()

    # 保存混淆矩阵数据
    cm = confusion_matrix(all_labels, all_preds)
    np.save('confusion_matrix.npy', cm)

    # 绘制混淆矩阵
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'

    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=np.arange(1, 7))  # 显示1-6
    disp.plot(cmap='Blues', ax=plt.gca(), values_format='d')
    plt.title('Confusion Matrix', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('confusion_matrix.png', dpi=600, bbox_inches='tight')
    plt.close()

    return all_labels, all_preds

test_loader = DataLoader(
    SpectrogramDataset('小波同步压缩变换时频/test_img', test_transform),
    batch_size=32,
    shuffle=False
)
true_labels, pred_labels = final_evaluation(model, test_loader)

# 打印评估指标
test_acc = (np.array(pred_labels) == np.array(true_labels)).mean()
precision = precision_score(true_labels, pred_labels, average='macro')  # 使用宏平均处理多分类
recall = recall_score(true_labels, pred_labels, average='macro')
f1 = f1_score(true_labels, pred_labels, average='macro')

print(f"Test Accuracy: {test_acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# print(f"Test Accuracy: {np.mean(np.array(true_labels) == np.array(pred_labels)):.4f}")
print("Confusion Matrix and t-SNE visualization saved.")
