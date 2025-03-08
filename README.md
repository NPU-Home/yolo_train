# YOLO 目标检测训练项目

> [!NOTE]
> 以下内容大体由Github Copilot生成，(仅供参考)

这是一个基于 [Ultralytics YOLO](https://docs.ultralytics.com/) 的目标检测模型训练项目，提供了完整的数据集处理、划分、训练和验证流程。

## 项目结构

```bash
yolo_train/
├── utils/                      # 工具函数
│   ├── __init__.py             # 初始化文件
│   └── split_dataset.py        # 数据集划分工具
├── dataset/                    # 原始数据集
│   ├── images/                 # 图像文件
│   ├── labels/                 # 标签文件
│   └── labels.txt              # 类别标签
├── outputs/                    # 输出目录
│   ├── dataset.yaml            # 生成的数据集配置
│   └── dataset/                # 划分后的数据集
├── runs/                       # 训练运行目录
│   └── detect/                 # 检测结果
├── yolo_train.py               # 训练主脚本
├── yolov8x.pt                  # YOLOv8预训练模型(x版本)
├── yolov8l.pt                  # YOLOv8预训练模型(l版本)
├── yolov8m.pt                  # YOLOv8预训练模型(m版本)
├── yolov8s.pt                  # YOLOv8预训练模型(s版本)
├── yolov8n.pt                  # YOLOv8预训练模型(n版本)
└── README.md                   # 项目说明文档
```

> [!NOTE]
> 以上项目结构中包含一些git仓库未提交的文件，为这些文件的建议放置路径

## 数据集格式

项目使用YOLO格式的数据集，处理前的结构应如下：

```bash
dataset/
├── images/                     # 图像目录
│   ├── 00001.jpg
│   ├── 00002.jpg
│   └── ...
├── labels/                     # 标签目录(YOLO格式)
│   ├── 00001.txt
│   ├── 00002.txt
│   └── ...
└── labels.txt                  # 类别定义文件
```

### 标签格式

- 每个`.txt`文件对应一张图片，包含该图片中的所有目标
- 每行代表一个目标，格式为：`class_id x_center y_center width height`
- 所有值都是相对于图像宽高的归一化坐标(0-1之间)

## 环境配置

### 依赖库

```bash
pip install ultralytics sklearn pyyaml
```

## 使用说明

### 数据集划分

使用`split_dataset.py`中的函数可以将数据集划分为训练集、验证集和测试集，示例如下：

```python
from utils import split_dataset, make_dataset_yaml

# 按照7:2:1的比例划分数据集
output_dataset_path, splits, classes = split_dataset(
    split_ratio='7:2:1',
    dataset_dir='mydataset'
)

# 生成YAML配置文件
make_dataset_yaml(
    dataset_path=output_dataset_path,
    splits=splits,
    classes=classes
)
```

### 模型训练

使用`yolo_train.py`脚本进行模型训练：

```bash
python yolo_train.py
```

该脚本会：
1. 根据需要划分数据集（如果SPLIT_DATASET=True）
2. 使用多个不同大小的YOLOv8模型进行训练
3. 在训练完成后进行验证（如果VAL=True）
4. 将最佳权重复制到项目根目录（如果COPY_TO_ROOT=True）

## 工具模块说明

### split_dataset.py

数据集划分工具，提供以下功能：
- `split_dataset()`: 将数据集划分为训练集、验证集和可选的测试集
- `make_dataset_yaml()`: 生成YOLOv8格式的数据集配置YAML文件

### 参数说明

- `split_ratio`: 划分比例，格式为"7:2:1"(训练:验证:测试)或"8:2"(训练:验证)
- `dataset_dir`: 数据集目录路径，相对于项目根目录
- `output_dirname`: 划分后数据集的输出目录名
- `copy_from_source`: 是否复制源文件（否则移动）

## 示例

1. 准备好YOLO格式的数据集，放在设置的目录下(如`mydataset/`)
2. 运行`yolo_train.py`进行模型训练
3. 在detect查看训练结果
4. 使用生成的最佳模型权重进行推理
