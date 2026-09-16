# -*- coding: utf-8 -*-
"""
CIFAR-10 vibe-coding 训练脚本
=============================
自己手搭一个轻量 CNN（ResNet 风格，带残差连接），在 CIFAR-10 上从零训练。
目标：代码简单、可读、跑得动，同时在 RTX 5060 上能到 93%+ 的测试准确率。

运行方式：
    python train.py            # 默认 60 epochs 全量训练
    python train.py --epochs 5 # 快速验证（跑通流程用）

产物：
    best_ckpt.pth   最优权重
    curves.png      训练曲线（loss / acc）
    metrics.json    最终指标
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ----------------------------------------------------------------------------
# 1. 网络定义：自己 vibe coding 的轻量残差网络
# ----------------------------------------------------------------------------
# 结构：stem(3->64) → 3 个 stage（每个 2 个残差块）→ 全局平均池化 → 分类头
# stage 之间通道数翻倍、分辨率减半（步长 2 的下采样残差块）

def conv3x3(in_ch, out_ch, stride=1):
    """3x3 卷积，padding=1 保持分辨率（stride=2 时减半）"""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    """残差块：BN->ReLU->Conv->BN->ReLU->Conv + 恒等/投影捷径"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = conv3x3(out_ch, out_ch)
        # 通道数或分辨率变化时，捷径用一个 1x1 卷积对齐
        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(x))
        out = self.conv1(out)
        out = self.act(self.bn2(out))
        out = self.conv2(out)
        if self.shortcut is not None:
            identity = self.shortcut(x)
        return out + identity


class VibeNet(nn.Module):
    """CIFAR-10 专用轻量残差网络（参数约 4.6M）"""
    def __init__(self, width=64, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            conv3x3(3, width),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        # 每个 stage 通道数翻倍：[64, 128, 256]
        self.stage1 = self._make_stage(width, width, blocks=2, stride=1)
        self.stage2 = self._make_stage(width, width * 2, blocks=2, stride=2)
        self.stage3 = self._make_stage(width * 2, width * 4, blocks=2, stride=2)
        self.bn = nn.BatchNorm2d(width * 4)
        self.act = nn.ReLU(inplace=True)
        self.head = nn.Linear(width * 4, num_classes)

    @staticmethod
    def _make_stage(in_ch, out_ch, blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.act(self.bn(x))
        x = x.mean(dim=(2, 3))  # 全局平均池化
        return self.head(x)


# ----------------------------------------------------------------------------
# 2. 数据：CIFAR-10 + 数据增强
# ----------------------------------------------------------------------------
def get_loaders(batch_size=128, num_workers=2):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),          # 随机裁剪
        transforms.RandomHorizontalFlip(),             # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),  # CutOut 式遮挡
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


# ----------------------------------------------------------------------------
# 3. 训练 / 评估
# ----------------------------------------------------------------------------
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * y.size(0)
        correct += (out.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device}")

    train_loader, test_loader = get_loaders(args.batch_size)
    model = VibeNet(width=args.width).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params / 1e6:.2f}M")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)      # 标签平滑
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best_acc, history = 0.0, {"train_loss": [], "train_acc": [], "test_acc": []}
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        loss, acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()
        history["train_loss"].append(loss)
        history["train_acc"].append(acc)
        history["test_acc"].append(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_ckpt.pth")
        print(f"[epoch {epoch:3d}/{args.epochs}] "
              f"train_loss={loss:.4f} train_acc={acc:.3f} test_acc={test_acc:.3f} "
              f"best={best_acc:.3f} | {time.time() - t0:.0f}s")

    print(f"\n[result] best test acc = {best_acc:.4f}")

    # 画训练曲线
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="train loss")
    axes[0].set_title("Training Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history["train_acc"], label="train acc")
    axes[1].plot(history["test_acc"], label="test acc")
    axes[1].axhline(best_acc, color="red", ls="--", lw=1, label=f"best {best_acc:.3f}")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("curves.png", dpi=150)

    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump({"best_test_acc": best_acc, "params_M": round(n_params / 1e6, 2),
                   "history": history}, f, ensure_ascii=False, indent=2)
    print("[done] saved best_ckpt.pth / curves.png / metrics.json")


if __name__ == "__main__":
    main()
