"""
数据集划分工具

该模块提供了将YOLO格式数据集划分为训练集、验证集和测试集的功能，
并生成相应的YAML配置文件供YOLOv8训练使用。
"""
from sklearn.model_selection import train_test_split
import glob
import shutil
import os
import yaml

# 当前脚本所在目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 项目根目录
ROOT_DIR = os.path.normpath(os.path.join(CURRENT_DIR, '..'))
# 输出目录
OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs')

# 包含测试集的三路划分
TRAIN_VAL_TEST = ('train', 'val', 'test')
# 不包含测试集的双路划分
TRAIN_VAL = ('train', 'val')

def split_dataset(
    split_ratio: str,
    dataset_dir: str,
    output_dirname: str = 'dataset',
    copy_from_source: bool = True,
) -> tuple[str, tuple[str, str, str] | tuple[str, str], list]:
    """
    将数据集划分为训练集、验证集和可选的测试集
    
    参数:
        split_ratio (str): 划分比例，格式为"7:2:1"(训练:验证:测试)或"8:2"(训练:验证)
        dataset_dir (str): 数据集目录路径，相对于项目根目录
        output_dirname (str): 划分后数据集的输出目录名，默认为'dataset'
        copy_from_source (bool): 是否复制源文件（而非移动），默认为True
        
    返回:
        tuple[str, tuple, list]: 包含以下元素的元组：
            - 输出数据集路径
            - 数据集划分类型元组(train,val,test)或(train,val)
            - 类别列表
    """
    # 解析比例字符串为整数列表
    split_sizes = list(map(int, split_ratio.split(':')))
    assert len(split_sizes) in [2, 3], "`split_ratio` should be like '7:2:1' or '8:2'"
    
    # 获取所有图像和标签文件路径
    imgs_rest = glob.glob(os.path.join(ROOT_DIR, dataset_dir, 'images', '*.jpg'))
    txts_rest = glob.glob(os.path.join(ROOT_DIR, dataset_dir, 'labels', '*.txt'))
    assert len(imgs_rest) == len(txts_rest), "Number of images and labels should be same"
    
    # 准备输出目录
    output_dataset_path = os.path.join(OUTPUT_DIR, output_dirname)
    os.makedirs(os.path.join(output_dataset_path), exist_ok=True)
    shutil.rmtree(os.path.join(output_dataset_path), ignore_errors=True)
    
    # 创建子目录结构：根据划分类型创建相应的images和labels子目录
    if len(split_sizes) == 3:
        for split in TRAIN_VAL_TEST:
            os.makedirs(os.path.join(output_dataset_path, 'images', split), exist_ok=False)
            os.makedirs(os.path.join(output_dataset_path, 'labels', split), exist_ok=False)
    else:
        for split in TRAIN_VAL:
            os.makedirs(os.path.join(output_dataset_path, 'images', split), exist_ok=False)
            os.makedirs(os.path.join(output_dataset_path, 'labels', split), exist_ok=False)
        
    # 根据参数设置操作类型：复制或移动
    cp_or_mv = shutil.copy if copy_from_source else shutil.move
        
    # 如果有三路划分，先拆分出测试集
    if len(split_sizes) == 3:
        imgs_rest, imgs_test, txts_rest, txts_test = train_test_split(
            imgs_rest, txts_rest, test_size=split_sizes[2] / sum(split_sizes), random_state=42
        )
        
        for img, txt in zip(imgs_test, txts_test):
            cp_or_mv(img, os.path.join(output_dataset_path, 'images', 'test'))
            cp_or_mv(txt, os.path.join(output_dataset_path, 'labels', 'test'))
    
    # 从剩余数据中划分训练集和验证集  
    imgs_train, imgs_val, txts_train, txts_val = train_test_split(
        imgs_rest, txts_rest, test_size=split_sizes[1] / sum(split_sizes[:2]), random_state=42
    )
    
    for img, txt in zip(imgs_val, txts_val):
        cp_or_mv(img, os.path.join(output_dataset_path, 'images', 'val'))
        cp_or_mv(txt, os.path.join(output_dataset_path, 'labels', 'val'))
        
    for img, txt in zip(imgs_train, txts_train):
        cp_or_mv(img, os.path.join(output_dataset_path, 'images', 'train'))
        cp_or_mv(txt, os.path.join(output_dataset_path, 'labels', 'train'))
        
    # 读取类别信息
    with open(os.path.join(ROOT_DIR, dataset_dir, 'labels.txt'), 'r') as f:
        classes = f.read().split("\n")
        
    return output_dataset_path, TRAIN_VAL_TEST if len(split_sizes) == 3 else TRAIN_VAL, classes

def make_dataset_yaml(
    dataset_path: str,
    splits: tuple[str, str, str] | tuple[str, str],
    classes: list,
    name: str = "dataset.yaml",
    copy_to_root: bool = True,
):
    """
    创建YOLOv8格式的数据集配置YAML文件
    
    参数:
        dataset_path (str): 数据集路径
        splits (tuple): 数据集划分类型，如('train','val','test')或('train','val')
        classes (list): 类别名称列表
        name (str): YAML文件名，默认为'dataset.yaml'
        copy_to_root (bool): 是否将YAML文件复制到项目根目录，默认为True
    """
    # 确保文件名有.yaml后缀
    if not name.endswith('.yaml'):
        name += '.yaml'
    filename = os.path.join(OUTPUT_DIR, name)
    
    # 创建YAML配置字典
    dataset_yaml = {
        "path": dataset_path,  # 数据集根路径
        **{sp: f"images/{sp}" for sp in splits},  # 各分割子目录
        "names": {i: cls for i, cls in enumerate(classes)}  # 类别索引映射
    }
    
    # 写入YAML文件
    with open(filename, 'w') as f:
        yaml.dump(dataset_yaml, f, allow_unicode=True, sort_keys=False)
        
    # 如果需要，复制到根目录 
    if copy_to_root:
        shutil.copy(filename, os.path.join(ROOT_DIR, name))
