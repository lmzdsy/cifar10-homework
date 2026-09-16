# CIFAR-10 训练作业

自写轻量残差网络（VibeNet）在 CIFAR-10 上的图像分类训练。

## 结果

- **测试集准确率：95.36%**（超过 95% 目标）
- 模型参数：2.78M
- 训练：60 epochs，约 35 分钟（RTX 5060，AMP 混合精度）

## 网络结构

手写 ResNet 风格 CNN：stem（3→64）+ 3 个 stage（每 stage 2 个残差块，通道 64/128/256）+ 全局平均池化 + 分类头。残差块采用 BN→ReLU→Conv→BN→ReLU→Conv 结构，通道/分辨率变化时用 1×1 卷积对齐捷径。

## 训练技巧

- 数据增强：随机裁剪 + 随机水平翻转 + 随机擦除（CutOut 式）
- 标签平滑 0.1
- 余弦退火学习率（SGD, lr=0.1, momentum=0.9, weight_decay=5e-4）
- AMP 混合精度训练

## 运行

```bash
pip install torch torchvision matplotlib
python train.py --epochs 60
```

首次运行会自动下载 CIFAR-10 数据集（约 170MB）到 `./data`。

## 文件

- `train.py` — 主训练脚本（VibeNet，95.36%）
- `train_wrn.py` — 升级版（WideResNet-28-4 + CutOut + MixUp，目标 96%+）
- `best_ckpt_vibenet.pth` — 最优权重
- `curves_vibenet.png` — 训练曲线
- `metrics_vibenet.json` — 全部 60 epoch 指标
