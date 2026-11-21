# Waste Image Classification with Limited Training Data

用 RealWaste 的五个类别，比较「从零训练的小 CNN」和「冻结骨干的 ImageNet ResNet-18」。训练图只取每类 10、50、200 张，并同时比较有无数据增强。主指标是测试集 Macro-F1。每个条件用种子 0、1、2 重复三次。

测试集在 36 次训练全部结束后才使用一次。下表是三次的平均值 ± 样本标准差。

| 每类训练张数 | 小 CNN，无增强 | 小 CNN，有增强 | ResNet-18，无增强 | ResNet-18，有增强 |
| --- | ---: | ---: | ---: | ---: |
| 10 | 0.408 ± 0.164 | 0.336 ± 0.112 | 0.668 ± 0.011 | 0.665 ± 0.017 |
| 50 | 0.673 ± 0.041 | 0.598 ± 0.079 | 0.794 ± 0.024 | 0.767 ± 0.005 |
| 200 | 0.769 ± 0.023 | 0.717 ± 0.035 | 0.849 ± 0.005 | 0.841 ± 0.004 |

训练图变多时两个模型都变好。冻结的 ResNet-18 在每个条件下都更高，数据越少优势越大。这里使用的增强没有提高 Macro-F1。

## 数据

只用 RealWaste 的五个类别：Food Organics、Glass、Metal、Paper、Plastic，共 3042 张，原图均为 524×524。读入模型时缩放到 224×224，并用 ImageNet 的均值和标准差做归一化。没有做近重复图片清理。

图片放在：

```text
data/raw/Food Organics/
data/raw/Glass/
data/raw/Metal/
data/raw/Paper/
data/raw/Plastic/
```

来源与许可见 `data/raw/SOURCE.txt`。

> Sam Single, Saeid Iranmanesh, and Raad Raad. RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning. *Information*, 14(12), 633, 2023. https://doi.org/10.3390/info14120633

数据集许可证是 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)。公开这些图片时需要保留署名、非商业使用，并以相同方式共享。

## 划分和种子

种子 42 只负责划分。每个类别按文件名排序后，用这个种子打乱，再切成大约 70% 训练、10% 验证、20% 测试。结果是训练 2130 张、验证 304 张、测试 608 张，写在 `splits/split.csv`。验证集和测试集始终用全量。

种子 0、1、2 各从训练集抽一份嵌套子集，名单在 `splits/subsets.json`。同一份打乱结果里，每类前 10 张包含在前 50 张里，前 50 张包含在前 200 张里。训练时 `--seed` 同时决定三件事：用哪一份子集、新层如何初始化、每个 epoch 的打乱顺序和增强顺序。

## 模型与训练

- 小 CNN：三个 Conv 3×3–BN–ReLU–MaxPool 块，通道 32/64/128，全局平均池化后接 `Linear(128 → 5)`，全部从零训练。学习率 `1e-3`。
- ResNet-18：使用 `IMAGENET1K_V1` 权重。骨干冻结，BatchNorm 保持 ImageNet 的统计量，只训练新的 `Linear(512 → 5)`。学习率 `1e-2`。
- 优化器 AdamW，权重衰减 `1e-4`，批大小 32，最多 40 个 epoch。验证集 Macro-F1 连续 8 个 epoch 没有提高就停止，并恢复最好的权重。
- 增强只加在训练图上：水平翻转、±15° 旋转、亮度抖动 0.2。验证和测试不做增强。

Macro-F1 是五个类别 F1 的简单平均，每一类权重相同。

## 环境

在 Python 3.12 上训练和评估，使用 PyTorch 2.14（CUDA 12.6）和对应的 torchvision。还需要 NumPy、Pillow、Matplotlib。有 NVIDIA GPU 时训练和评估会自动使用 CUDA。

## 复现

在项目根目录执行。

```bash
python src/prepare_data.py
python src/train.py --model cnn --seed 0 --images-per-class 50
python src/train.py --model resnet --seed 0 --images-per-class 200 --augment
python src/evaluate.py
python src/plot_results.py
```

完整实验是 36 次：模型 `cnn` 或 `resnet`，`--images-per-class` 取 10、50、200，`--seed` 取 0、1、2，增强与否各一次。加 `--augment` 表示开启增强。每次权重保存为：

```text
results/{model}_seed{seed}_n{数量}_{aug|noaug}.pt
```

`src/evaluate.py` 要求 `results` 里正好有这 36 个文件。它在固定测试集上各评一次，写出：

- `results/test_scores.csv`：每次的测试 Macro-F1 和各类 F1
- `results/test_summary.csv`：12 个条件的均值和标准差
- `results/test_confusion.json`：每次的混淆矩阵

`src/plot_results.py` 把报告用的图写到 `figures/`：

| 文件 | 内容 |
| --- | --- |
| `test_macro_f1.png` | 测试 Macro-F1 随每类张数的变化 |
| `per_class_f1.png` | 每类 200 张、无增强时各类的测试 F1 |
| `confusion_matrices.png` | 同一条件下、三个种子合计后的混淆比例 |
| `cnn_val_curves.png` | 无增强小 CNN 在每类 10 张和 200 张时的验证曲线 |
| `split_counts.png` | 各类的训练、验证、测试张数 |
| `sample_images.png` | 测试集中每类 4 张样例 |

## 目录

```text
src/prepare_data.py    固定划分和嵌套子集
src/dataset.py         读取图片并缩放到 224
src/models.py          小 CNN 和冻结的 ResNet-18
src/train.py           训练；不读取测试集
src/evaluate.py        36 个权重的一次测试集评估
src/plot_results.py    报告图
splits/                split.csv 和 subsets.json
results/               权重、训练日志和测试分数
figures/               报告图
```
