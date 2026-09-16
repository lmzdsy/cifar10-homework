# -*- coding: utf-8 -*-
"""
CIFAR-10 vibe-coding 训练脚本（95%+ 目标）
==========================================
在 train.py 基线网络（VibeNet，约 90%）基础上升级：
  - 手写 WideResNet-28-4 风格网络（约 5.9M 参数）
  - CutOut 遮挡 + MixUp 混合增强
  - 标签平滑 + 余弦退火 + Nesterov SGD
  - AMP 混合精度加速

运行方式：
    python train_wrn.py                     # 默认 120 epochs
    python train_wrn.py --epochs 20         # 快速验证

产物：
    best_ckpt.pth   最优权重
    curves.png      训练曲线（loss / acc）
    metrics.json    最终指标
"""
import argparse
import json
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ----------------------------------------------------------------------------
# 1. 网络定义：手写 WideResNet 风格（加宽残差块，PreActivation + Dropout）
# ----------------------------------------------------------------------------
class WideBasic(nn.Module):
    """加宽残差块：BN->ReLU->Conv->Dropout->BN->ReLU->Conv + 1x1 投影捷径"""
    def __init__(self, in_ch, out_ch, stride=1, drop_rate=0.3):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout2d(drop_rate) if drop_rate > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        # 通道/分辨率变化时用 1x1 卷积对齐捷径
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if (stride != 1 or in_ch != out_ch) else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = self.drop(F.relu(self.bn2(out)))
        out = self.conv2(out)
        return out + identity


class VibeWideNet(nn.Module):
    """WideResNet-28-4：stem(16ch) → 3 个 stage(通道 64/128/256) → GAP → 分类头"""
    def __init__(self, depth=28, widen=4, num_classes=10, drop_rate=0.3):
        super().__init__()
        n = (depth - 4) // 6                      # 每个 stage 的残差块数（28→4）
        k = widen                                 # 加宽因子
        channels = [16, 16 * k, 32 * k, 64 * k]
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.stage1 = self._make_stage(16, channels[1], n, stride=1, drop_rate=drop_rate)
        self.stage2 = self._make_stage(channels[1], channels[2], n, stride=2, drop_rate=drop_rate)
        self.stage3 = self._make_stage(channels[2], channels[3], n, stride=2, drop_rate=drop_rate)
        self.bn = nn.BatchNorm2d(channels[3])
        self.head = nn.Linear(channels[3], num_classes)

    @staticmethod
    def _make_stage(in_ch, out_ch, blocks, stride, drop_rate):
        layers = [WideBasic(in_ch, out_ch, stride, drop_rate)]
        for _ in range(1, blocks):
            layers.append(WideBasic(out_ch, out_ch, 1, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.relu(self.bn(x))
        x = x.mean(dim=(2, 3))
        return self.head(x)


# ----------------------------------------------------------------------------
# 2. 数据增强：CutOut 遮挡 + MixUp 混合
# ----------------------------------------------------------------------------
class Cutout:
    """随机挖掉一个方形区域（置 0），强迫网络用上下文信息识别"""
    def __init__(self, n_holes=1, length=16):
        self.n_holes, self.length = n_holes, length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        for _ in range(self.n_holes):
            cy, cx = random.randint(0, h), random.randint(0, w)
            y0 = max(0, cy - self.length // 2); y1 = min(h, cy + self.length // 2)
            x0 = max(0, cx - self.length // 2); x1 = min(w, cx + self.length // 2)
            img[:, y0:y1, x0:x1] = 0
        return img


def get_loaders(batch_size=256, num_workers=2):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        Cutout(n_holes=1, length=16),
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
# 3. MixUp 混合
# ----------------------------------------------------------------------------
def mixup_data(x, y, alpha=0.2):
    """把两个样本按 λ 比例线性混合，标签也按 λ 混合"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ----------------------------------------------------------------------------
# 4. 训练 / 评估
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


def evaluate_tta(model, loader, device):
    """测试时增强：原图 + 水平翻转 的 logits 取平均，白捡 0.3~0.5%"""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x) + model(torch.flip(x, dims=[3]))
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def train_epoch(model, loader, criterion, optimizer, scaler, device, mixup_alpha):
    model.train()
    running_loss = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            if mixup_alpha > 0:
                x, y_a, y_b, lam = mixup_data(x, y, mixup_alpha)
                out = model(x)
                loss = mixup_criterion(criterion, out, y_a, y_b, lam)
            else:
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
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--widen", type=int, default=4)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device}")

    train_loader, test_loader = get_loaders(args.batch_size)
    model = VibeWideNet(depth=args.depth, widen=args.widen).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] VibeWideNet-{args.depth}-{args.widen} params={n_params / 1e6:.2f}M")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best_acc, history = 0.0, {"train_loss": [], "train_acc": [], "test_acc": []}
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        loss, acc = train_epoch(model, train_loader, criterion, optimizer, scaler,
                                device, args.mixup_alpha)
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

    # 用最优权重做最终评估（含 TTA）
    model.load_state_dict(torch.load("best_ckpt.pth", map_location=device, weights_only=True))
    final_acc = evaluate(model, test_loader, device)
    final_tta = evaluate_tta(model, test_loader, device)
    print(f"[final] test acc = {final_acc:.4f} | with flip-TTA = {final_tta:.4f}")

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
        json.dump({"best_test_acc": best_acc, "final_test_acc": final_acc,
                   "final_tta_acc": final_tta, "params_M": round(n_params / 1e6, 2),
                   "history": history}, f, ensure_ascii=False, indent=2)
    print("[done] saved best_ckpt.pth / curves.png / metrics.json")


if __name__ == "__main__":
    main()
