# CIFAR-10 训练作业

自写轻量残差网络在 CIFAR-10 上的图像分类训练。

## 结果

| 模型 | 参数 | 测试准确率 | 说明 |
|---|---|---|---|
| **VibeWideNet-28-4** | 5.85M | **95.76%**（TTA 后 95.93%） | 最终提交版本 |
| VibeNet | 2.78M | 95.36% | 轻量基线 |

训练：120 epochs（WRN）/ 60 epochs（VibeNet），RTX 5060 + AMP 混合精度。

## 网络结构

**VibeNet**：手写 ResNet 风格 CNN。stem（3→64）+ 3 个 stage（每 stage 2 个残差块，通道 64/128/256）+ 全局平均池化 + 分类头。

**VibeWideNet**：手写 WideResNet-28-4。加宽残差块（PreActivation + Dropout2d），stem（16 通道）+ 3 个 stage（通道 64/128/256）+ GAP + 分类头。

## 训练技巧

- 数据增强：随机裁剪 + 随机水平翻转 + 随机擦除（VibeNet）/ CutOut + MixUp（VibeWideNet）
- 标签平滑 0.1
- 余弦退火学习率（SGD, lr=0.1, momentum=0.9, weight_decay=5e-4, Nesterov）
- 测试时增强 TTA（水平翻转 logits 平均）
- AMP 混合精度训练

## 运行

```bash
pip install torch torchvision matplotlib
python train.py --epochs 60      # VibeNet，95.36%
python train_wrn.py --epochs 120 # VibeWideNet-28-4，95.76%
```

首次运行会自动下载 CIFAR-10 数据集（约 170MB）到 `./data`。

## 文件

- `train.py` — VibeNet 训练脚本（95.36%）
- `train_wrn.py` — VibeWideNet-28-4 训练脚本（95.76%，TTA 95.93%）
- `best_ckpt_wrn.pth` / `best_ckpt_vibenet.pth` — 最优权重
- `curves_wrn.png` / `curves_vibenet.png` — 训练曲线
- `metrics_wrn.json` / `metrics_vibenet.json` — 全部 epoch 指标
