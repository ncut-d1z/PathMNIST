import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import medmnist
from medmnist import INFO
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import time
import os

# ==========================================
# 1. 配置与参数
# ==========================================
class Config:
    DATA_FLAG = 'pathmnist'
    BATCH_SIZE = 128
    NUM_EPOCHS = 20  # 如果需要更高精度，可适当增加到30-50
    LEARNING_RATE = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = INFO[DATA_FLAG]
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    image_size = 28

print(f"使用设备: {Config.DEVICE}")
print(f"数据集: {Config.DATA_FLAG}, 类别数: {Config.n_classes}")

# ==========================================
# 2. 数据预处理与加载
# ==========================================
def get_dataloaders():
    # 数据增强：对医学图像来说，旋转和翻转通常是安全的且能显著提高性能
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5]) # PathMNIST 推荐归一化
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    DataClass = getattr(medmnist, Config.info['python_class'])

    train_dataset = DataClass(split='train', transform=train_transform, download=True)
    val_dataset = DataClass(split='val', transform=test_transform, download=True)
    test_dataset = DataClass(split='test', transform=test_transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset

# 可视化函数
def visualize_data(dataset):
    print("正在生成数据可视化...")
    plt.figure(figsize=(10, 10))
    for i in range(25):
        img, label = dataset[i]
        img = img.permute(1, 2, 0).numpy() * 0.5 + 0.5 # 反归一化
        plt.subplot(5, 5, i + 1)
        plt.imshow(img)
        plt.title(f"Label: {label[0]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # 类别分布分析
    targets = dataset.labels
    plt.figure(figsize=(8, 5))
    plt.hist(targets, bins=np.arange(Config.n_classes+1)-0.5, rwidth=0.8, color='skyblue', edgecolor='black')
    plt.title("Class Distribution in Training Set")
    plt.xlabel("Class Label")
    plt.ylabel("Frequency")
    plt.xticks(range(Config.n_classes))
    plt.show()

# ==========================================
# 3. 模型定义
# ==========================================

# --- 模型 A: ResNet-18 (针对 28x28 优化) ---
class ResNet18_28x28(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18_28x28, self).__init__()
        # 加载标准 ResNet18
        self.model = models.resnet18(pretrained=False)

        # 修改第一层卷积：适应小尺寸输入
        # 原版是 7x7 conv, stride 2 -> 适合 224x224
        # 改为 3x3 conv, stride 1 -> 适合 28x28 (参考 CIFAR-10 的改法)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # 移除第一层的 MaxPool，避免信息过早丢失
        self.model.maxpool = nn.Identity()

        # 修改全连接层
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# --- 模型 B: Vision Transformer (ViT) ---
# 由于 28x28 图像太小，直接用 standard ViT (patch size 16) 会导致 patch 只有 2x2 个，效果很差。
# 这里实现一个轻量级的 ViT，使用 patch size = 4
class MiniViT(nn.Module):
    def __init__(self, image_size=28, patch_size=4, num_classes=9, dim=128, depth=6, heads=8, mlp_dim=256, channels=3):
        super().__init__()
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = channels * patch_size * patch_size
        self.dim = dim

        self.patch_to_embedding = nn.Sequential(
            nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2) # [B, dim, num_patches]
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.patch_to_embedding(x) # [B, dim, num_patches]
        x = x.transpose(1, 2) # [B, num_patches, dim]

        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        x = self.transformer(x)

        cls_output = x[:, 0]
        return self.mlp_head(cls_output)

# ==========================================
# 4. 训练与评估流程
# ==========================================
def train_model(model, train_loader, val_loader, model_name="Model"):
    criterion = nn.CrossEntropyLoss()
    # 使用 AdamW 优化器，通常比 SGD 收敛更快更好
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)

    train_acc_history = []
    val_acc_history = []
    train_loss_history = []

    print(f"\n开始训练 {model_name} ...")
    start_time = time.time()

    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            labels = labels.squeeze().long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        epoch_acc = 100. * correct / total
        epoch_loss = running_loss / len(train_loader)
        train_acc_history.append(epoch_acc)
        train_loss_history.append(epoch_loss)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
                labels = labels.squeeze().long()
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        val_acc_history.append(val_acc)

        print(f"Epoch [{epoch+1}/{Config.NUM_EPOCHS}] Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    end_time = time.time()
    print(f"{model_name} 训练完成。耗时: {end_time - start_time:.2f} 秒")

    return {
        'train_acc': train_acc_history,
        'val_acc': val_acc_history,
        'loss': train_loss_history,
        'model': model
    }

def evaluate_model(model, test_loader, model_name="Model"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            labels = labels.squeeze().long()
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算指标
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    print(f"\n=== {model_name} 测试集评估结果 ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(Config.n_classes))
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap='Blues', ax=ax)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

    return acc, f1

# ==========================================
# 5. 主程序执行
# ==========================================
if __name__ == "__main__":
    # 1. 准备数据
    train_loader, val_loader, test_loader, train_dataset = get_dataloaders()
    visualize_data(train_dataset)

    # 2. 训练 ResNet
    resnet_model = ResNet18_28x28(num_classes=Config.n_classes).to(Config.DEVICE)
    resnet_results = train_model(resnet_model, train_loader, val_loader, model_name="ResNet-18")

    # 3. 训练 ViT
    vit_model = MiniViT(num_classes=Config.n_classes).to(Config.DEVICE)
    vit_results = train_model(vit_model, train_loader, val_loader, model_name="ViT")

    # 4. 评估对比
    print("\n正在生成对比结果...")
    res_acc, res_f1 = evaluate_model(resnet_results['model'], test_loader, "ResNet-18")
    vit_acc, vit_f1 = evaluate_model(vit_results['model'], test_loader, "ViT")

    # 5. 绘制训练曲线对比
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(resnet_results['train_acc'], label='ResNet Train')
    plt.plot(resnet_results['val_acc'], label='ResNet Val')
    plt.plot(vit_results['train_acc'], label='ViT Train', linestyle='--')
    plt.plot(vit_results['val_acc'], label='ViT Val', linestyle='--')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(resnet_results['loss'], label='ResNet Loss')
    plt.plot(vit_results['loss'], label='ViT Loss', linestyle='--')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.show()
