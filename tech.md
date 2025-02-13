以下为针对该项目所有核心文件的**详细技术方案说明文档**。文档将从项目整体结构、数据处理流程、模型结构与原理、训练与推断流程、主要脚本功能等方面进行详细说明，帮助读者快速了解并使用此项目。

---

## 1. 项目简介

本项目实现了两种基于 Transformer 思想并结合卷积运算的轻量化人类活动识别（HAR）模型，分别是：

- **HART**: 通过传感器信号分块 + 轻量化多头注意力机制 + 卷积混合的方式，对加速度计和陀螺仪数据进行联合建模。
- **MobileHART**: 进一步在 HART 基础上引入 MobileNet/MobileViT 风格的骨干结构，更加面向移动设备端的模型。

两种模型针对加速度计和陀螺仪六通道（3 轴加速度 + 3 轴陀螺仪）的时间序列数据进行特征提取和分类。项目以 TensorFlow Keras 2.x（tested on TF 2.10.1）为深度学习框架，使用 Python 3.7+ 版本进行开发。

项目内附了一整套从数据获取与预处理到模型训练、可视化与评估、导出推断的端到端实现示例。

---

## 2. 目录结构与文件说明

以下是项目的主要目录、文件及其作用概览（参考 `<file_map>`）：

```
Lightweight-Transformer-Models-For-HAR-on-Mobile-Devices
├── checkpoints                 # （可选）保存训练过程中的checkpoints
├── datasets                    # 数据预处理与下载脚本存放目录
│   └── dataset                # 原始或下载后的数据存放位置
│       └── UCI HAR Dataset    # UCI官方数据集下载目录示例
├── examples                   # 额外示例或可视化脚本（如果有）
├── HART_Results               # 训练后生成的结果、模型权重、可视化输出等
│   └── ...
├── inference_benchmarks       # （可选）推断Benchmark脚本与相关资源
├── UCI                        # （可选）可能是单独对UCI做测试的结果文件夹
└── *.py / *.md / requirements.txt 等核心文件
```

在上述结构中，核心的 Python 文件及其作用如下：

1. **`main.py`**  
   - 项目主入口脚本。  
   - 主要功能：根据用户指定的超参数（例如模型类型、数据集名称、batch 大小、训练轮数、是否做位置/设备划分等），完成数据加载、模型创建与编译、模型训练和评估、可视化以及模型保存等流程。  
   - 内部会调用 `model.py` 中定义的 HART / MobileHART 结构，也会调用 `utils.py` 来做一些通用的功能，如数据加载、可视化等。

2. **数据集预处理脚本**（位于 `datasets` 文件夹）  
   - **`DATA_UCI.py`**: 负责下载并预处理 [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)。包括自动下载 ZIP 并解压，切分数据为训练和测试，进行标准化，最终将标准化后的数据存成 `.hkl` 格式。  
   - **`DATA_HHAR.py`**: 负责下载并预处理 [HHAR (Heterogeneity Activity Recognition) Dataset](http://archive.ics.uci.edu/ml/datasets/Heterogeneity+Activity+Recognition)。该数据集有来自不同设备（nexus4, lgwatch, s3, s3mini, gear, samsungold）的传感器数据，可进行设备差异分析。  
   - **`DATA_MotionSense.py`**: 负责下载并预处理 [MotionSense](https://github.com/mmalekzadeh/motion-sense) 数据集。  
   - **`DATA_REALWORLD.py`**: 负责下载并预处理 RealWorld2016 Dataset；该数据包含多个传感器位置（chest, forearm, head, shin, thigh, upperarm, waist）的数据，可用于不同身体部位位置的泛化分析。  
   - **`DATA_SHL.py`**: 负责下载并预处理 [SHL (Sussex-Huawei Locomotion) Dataset Preview](http://www.shl-dataset.org/)。  
   - 这些脚本主要功能均包括：  
     1. 自动下载（可选），如无法下载可手动把数据放到 `dataset` 目录。  
     2. 解压缩并读取原始文件；  
     3. 按窗口（通常 128 或 256长度）切分；  
     4. 对加速度计和陀螺仪数据进行标准化/去均值等；  
     5. 将处理好的数据按用户或设备或其他方式切分成训练/测试集，并存储为 `hkl` 格式。

3. **`model.py`**  
   - 定义了两大模型结构：**HART** 和 **MobileHART**（以及一些变体：mobileHART_XS, mobileHART_XXS 等）。  
   - 其中：
     - `HART(input_shape, activityCount, ...)`: 基于**传感器分块+LiteFormer+多头注意力**的轻量化 Transformer，用于 HAR。  
     - `mobileHART_XS(...)` 等函数：在局部特征提取中融合 MobileNet / MobileViT 风格，结合多头注意力完成 HAR。  
   - 该文件还定义了大量层级模块，包括 `SensorWiseMHA`, `liteFormer`, `DropPath`, `PatchEncoder`, `ClassToken` 等，使得模型具有可插拔和可定制化的特点。  
   - 同时提供了辅助函数如 `mlp()`, `mlp2()` 等，用于 MLP block 的构造。

4. **`utils.py`**  
   - 定义了一些通用的工具函数，例如：  
     - `loadDataset`: 根据所需数据集名称从预处理好的 `.hkl` 文件加载数据，并完成训练/测试集划分（使用了五折交叉方式但只取其中一折作为测试）。  
     - `plot_learningCurve`: 用于绘制训练过程的曲线。  
     - `projectTSNE` / `projectTSNEWithPosition`: 用于对学到的表征进行 t-SNE 降维可视化。  
     - `extract_intermediate_model_from_base_model`: 用于将某一层输出截断为新的模型，便于做特征可视化或提取。

5. **`README.md`**  
   - 项目自带的说明文档，介绍了使用方法、依赖库、引用链接等。

6. **`requirements.txt`**  
   - 记录了此项目所需的 Python 库以及对应版本约束。例如：`tensorflow>=2.10.1, numpy>=1.19.2, scikit-learn>=0.24.2` 等。

---

## 3. 环境与依赖

### 3.1 Python & TensorFlow 版本

- Python >= 3.7
- TensorFlow >= 2.10.1 （CPU 或 GPU 版本均可，推荐 GPU 训练）

### 3.2 其他主要依赖

- `numpy, pandas, scikit-learn, matplotlib, seaborn, hickle, scipy, resampy` 等
- 可通过 `pip install -r requirements.txt` 一次性安装。

若需要使用 GPU，则需确保本机安装合适的 CUDA (>= 11.2) 及 cuDNN 版本。

---

## 4. 数据预处理与加载流程

此项目在 `datasets` 文件夹下提供了一系列脚本，分别对应不同数据集。脚本内部流程大致相同：

1. **下载并解压**：  
   脚本中提供了 `download_url` 函数，会尝试从官方链接下载数据集。如果下载不成功，可以自行去官方链接下载并解压至 `datasets/dataset/extracted` 目录下。

2. **文件读取**：  
   使用 `pandas` 或原生 Python 读取 CSV、txt 文件。根据数据集格式可能会对文件进行拼接操作。

3. **固定窗口切分**：  
   将连续时间序列以 128 或 256 长度的滑动窗口（带一定 overlap）进行切分。  
   - 例如 `segmentData(accData, 128, 64)` 意味着窗口大小为 128，滑动步长为 64。

4. **标准化 / 去均值**：  
   计算加速度和陀螺仪分量的全局均值和标准差，然后做 Z-score 归一化。  
   通常 `(value - mean) / std`。

5. **存储为 hickle (hkl) 文件**：  
   为了后续使用方便，各脚本会将预处理结果(`X` 和 `y`)存到 `datasetStandardized/xxx` 路径下，对应如 `UserData{idx}.hkl` 和 `UserLabel{idx}.hkl`，或者 `trainX.hkl`、`testX.hkl`。

6. **后续加载**：  
   训练脚本 `main.py` 中会通过 `utils.py` 里 `loadDataset()` 函数，自动读取上述 `.hkl` 文件并按需求进行训练/验证/测试划分。

### 4.1 UCI Dataset

- 文件：`DATA_UCI.py`  
- 主要操作：  
  - 下载解压 `UCI HAR Dataset.zip`  
  - 读取 `train/` 及 `test/` 文件夹内的传感器信号  
  - 标准化并拆分成 `trainX.hkl`, `trainY.hkl`, `testX.hkl`, `testY.hkl`

### 4.2 HHAR Dataset

- 文件：`DATA_HHAR.py`  
- 主要操作：  
  - 下载解压 `heterogeneity activity recognition.zip`  
  - 对不同设备 (nexus4, lgwatch, s3, s3mini, gear, samsungold) 的数据合并，并按窗口切分。  
  - 输出每个“用户-设备”的数据 `UserData{i}.hkl` / `UserLabel{i}.hkl` 以及 `deviceIndex.hkl`。

### 4.3 MotionSense

- 文件：`DATA_MotionSense.py`  
- 下载自 [mmalekzadeh/motion-sense](https://github.com/mmalekzadeh/motion-sense)  
- 切分为 128 长度窗口，64 overlap。  
- 输出 `UserData{i}.hkl` 与 `UserLabel{i}.hkl`。

### 4.4 RealWorld

- 文件：`DATA_REALWORLD.py`  
- 下载并解压 Mannheims 公开的 realworld2016_dataset.zip  
- 数据中包含多个身体位置（如 chest, forearm, head, shin, thigh, upperarm, waist）  
- 在脚本中先对加速度 / 陀螺仪文件做拆分、对齐、重采样，再做 128 窗口  
- 输出 `clientsData.hkl`、`clientsLabel.hkl`，以 `(client, orientation)` 形式组织。

### 4.5 SHL Dataset

- 文件：`DATA_SHL.py`  
- 下载自 [SHL Dataset 预览版](http://www.shl-dataset.org/)  
- 同样进行解压、数据对齐、切分与标准化等操作  
- 输出 `clientsData.hkl`, `clientsLabel.hkl`

---

## 5. 模型结构与实现原理

### 5.1 HART (Human Activity Recognition Transformer)

**`HART()` 函数原型**：
```python
def HART(
    input_shape, activityCount, 
    projection_dim=192, patchSize=16, timeStep=16, 
    num_heads=3, filterAttentionHead=4, 
    convKernels=[3, 7, 15, 31, 31, 31],
    mlp_head_units=[1024], dropout_rate=0.3, useTokens=False
):
    ...
```

**核心思路**：
1. **SensorPatches**: 利用卷积核对加速度与陀螺仪做初步特征投影，得到 `[batch, patch_count, projection_dim]` 的patch嵌入序列。  
2. **ClassToken（可选）**: 若 `useTokens=True`，在每个序列首端加一个可训练的 `[CLS]` 向量。  
3. **LiteFormer + SensorWiseMHA**:  
   - LiteFormer(`liteFormer`)对输入序列在局部（acc / gyro中间部分）做深度可分离卷积形式的注意力操作（类似小卷积滤波权重），再**加**传感器自身的多头注意力 (`SensorWiseMHA`)。  
   - `SensorWiseMHA` 分别对 `[0:projectionQuarter]`（acc 通道）和 `[projectionQuarter + projectionHalf: projection_dim]`（gyro 通道）做 MHA；中间 `[projectionQuarter: projectionQuarter+projectionHalf]` 则是 LiteFormer 部分。  
4. **跳跃连接 + MLP**: 模块化地叠加多层。  
5. **Pool (GAP 或取 [CLS])**: 若有 token 则用 `[CLS]` 向量，否则用 GlobalAveragePooling1D。  
6. **MLP Head** 输出到活动类别。  

### 5.2 MobileHART

**`mobileHART_XS()` 函数**：
```python
def mobileHART_XS(
    input_shape, activityCount,
    projectionDims=[96,120,144],
    filterCount=[8,16,24,32,80,96,384],
    expansion_factor=4,
    mlp_head_units=[1024],
    dropout_rate=0.3
):
    ...
```

**核心思路**：
1. 借鉴 **MobileNetV2** 的 inverted residual block + 轻量化深度可分离卷积，进行初步下采样。  
2. 融合 **MobileViT** 的思路，将局部卷积特征转化为 Transformer block 进行全局上下文捕获，再通过卷积折叠回去。  
3. 对加速度与陀螺仪分支分别做局部卷积，然后在中间融合并引入 LiteFormer / MHA 机制。  
4. 最终做全局池化 + 全连接预测。  

相较于 HART，MobileHART 在前期特征提取采用了更多 MobileNet / MobileViT 的结构，能在移动端速度更快、占用更少。

---

## 6. 训练与评估流程

### 6.1 `main.py` 主要流程

1. **解析命令行参数**：  
   在 `main.py` 中，通过 `argparse` 读取如下超参数：
   - `--architecture`: 选择模型（`HART` 或 `MobileHART`）。
   - `--dataset`: 选择数据集（`UCI`, `HHAR`, `RealWorld`, `MotionSense`, `SHL`, 或 `COMBINED`）。
   - `--frame_length`, `--time_step`, `--projection_dim`, `--localEpoch`, `--batch_size` 等其他可配置项。
   - `--positionDevice`：若设置，可在 RealWorld 或 HHAR 中做某个位置 / 某个设备的留一测试。
2. **加载数据**：  
   调用 `utils.loadDataset(...)`，从 `datasetStandardized/` 下读取对应数据的 `.hkl` 文件，把它们按 70-10-20 或者五折之一做训练、验证、测试集等。  
   - 若指定了 `positionDevice`，脚本会将对应位置/设备的数据单独作为测试集，其余作为训练集。
3. **计算并保存预处理参数**：  
   - 计算训练集的加速度/陀螺仪 mean、std 并保存到 JSON 中，用于后续推断阶段的一致性。  
4. **构建并编译模型**：  
   - 如果 `architecture == "HART"`，则 `model_classifier = model.HART(...)`；  
   - 否则 `model_classifier = model.mobileHART_XS(...)`。  
   - 使用 Adam 优化器，学习率 `learningRate = 5e-3`（默认）等。
5. **训练**：  
   - 调用 `model.fit(...)`，并在验证集上做 early stopping / best checkpoint。  
   - 训练完后保存最优权重到 `HART_Results/...` 中。
6. **测试**：  
   - 使用最优权重在测试集评估准确率、F1 等指标。
7. **可视化**：
   - 绘制训练曲线到 `LearningAccuracy.svg` 和 `ModelLoss.svg`。  
   - 提取中间层进行 t-SNE 降维并保存散点图 `TSNE_Embeds`。
   - 使用 `SensorWiseMHA` 的注意力权重绘制注意力可视化图保存到 `attentionImages/` 文件夹。
8. **导出 TFLite**（可选）：
   - 在脚本末尾有一段 `tflite_model = converter.convert()`，会导出一个轻量级的 `.tflite` 模型文件，便于移动端部署。

### 6.2 训练过程中可视化与注意力图

在 `main.py` 中，示例性地对某些测试样本进行注意力可视化。  
- 调用了 `model_classifier.layers[...]` 提取最终的注意力权重，然后对时间序列作可视化叠加 highlight，不同程度阴影代表注意力大小。

### 6.3 留一位置 / 留一设备

若在 `RealWorld` 数据集中加上 `--positionDevice chest`，则表示将胸部数据作为测试集，其他位置作为训练集；  
若在 `HHAR` 数据集中加上 `--positionDevice nexus4`，则以 nexus4 设备的数据作为测试集，其他设备合并做训练。

---

## 7. 运行示例

以 **UCI** 数据集、HART 模型为例，默认参数：

```bash
python main.py --architecture HART --dataset UCI --localEpoch 200 --batch_size 64
```

若要在 **MotionSense** 数据集上跑 MobileHART：

```bash
python main.py --architecture MobileHART --dataset MotionSense --localEpoch 200 --batch_size 64
```

若要做**位置留一**（RealWorld）：

```bash
python main.py --architecture HART --dataset RealWorld --positionDevice chest --localEpoch 200 --batch_size 64
```

---

## 8. 结果与总结

- **HART** 和 **MobileHART** 在多个人体活动数据集上都展示了出色的精度和泛化性。  
- MobileHART 由于使用了类似 MobileNet 的结构，对移动端部署更加友好；HART 则具有相对简单且直接的混合 Transformer 结构，易于理解和二次开发。  
- 项目中也支持对注意力可视化（以对每个时间段的重要性进行可视化），以及在**位置**和**设备**层面的泛化测试。

**参考文献**（简要）：

1. Ek, S., Portet, F., & Lalanda, P. (2023). Transformer-based models to deal with heterogeneous environments in Human Activity Recognition. *Personal and Ubiquitous Computing*.
2. Ek, S., Portet, F., & Lalanda, P. (2022). Lightweight Transformers for Human Activity Recognition on Mobile Devices. *arXiv preprint* arXiv:2209.11750.

---

## 9. 总体建议与注意事项

1. **硬件建议**：  
   如果数据量较大（如 RealWorld 或 SHL），推荐使用 GPU 训练，否则训练可能会偏慢。
2. **数据对齐**：  
   某些数据集可能有缺失时间戳或不同采样率，需要在预处理脚本中仔细对齐和重采样。
3. **模型超参数**：  
   - `projection_dim`, `dropout_rate`, `timeStep`，`frameLength` 等皆可以根据实际需求做调整，需注意训练速度与效果平衡。
4. **部署**：  
   - 若要在移动端部署，建议使用 `model_classifier.tflite`，通过 TF Lite API 加载并推断。

---

## 10. 结语

本技术方案文档介绍了该 HAR-Transformer 项目的核心思路与实现细节，包括数据预处理、模型结构与核心函数、训练与评估流程以及可视化与部署方法。通过该项目，研究者或开发者可快速上手以 HART 或 MobileHART 进行人类活动识别，并可基于脚本进行自由扩展或与其他数据集进行对比。

如需进一步细节或深度定制，可从下列方向入手：

- **模型层面**：调整 `convKernels`、`num_heads`、`projection_dim` 等。
- **数据层面**：引入更多自定义数据集或特征工程方法。
- **可视化层面**：进一步研究注意力分布是否能解释特定活动的特征等。

希望此文档能帮助您更好地理解和使用该项目！