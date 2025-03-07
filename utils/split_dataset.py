from sklearn.model_selection import train_test_split
import glob
import shutil
import os
import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(CURRENT_DIR, '..'))
OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs')

TRAIN_VAL_TEST = ('train', 'val', 'test')
TRAIN_VAL = ('train', 'val')

def split_dataset(
    split_ratio: str,
    dataset_dir: str,
    output_dirname: str = 'dataset',
    copy_from_source: bool = True,
) -> tuple[str, tuple[str, str, str] | tuple[str, str], list]:
    
    split_sizes = list(map(int, split_ratio.split(':')))
    assert len(split_sizes) in [2, 3], "`split_ratio` should be like '7:2:1' or '8:2'"
    
    imgs_rest = glob.glob(os.path.join(ROOT_DIR, dataset_dir, 'images', '*.jpg'))
    txts_rest = glob.glob(os.path.join(ROOT_DIR, dataset_dir, 'labels', '*.txt'))
    assert len(imgs_rest) == len(txts_rest), "Number of images and labels should be same"
    
    output_dataset_path = os.path.join(OUTPUT_DIR, output_dirname)
    os.makedirs(os.path.join(output_dataset_path), exist_ok=True)
    shutil.rmtree(os.path.join(output_dataset_path), ignore_errors=True)
    
    if len(split_sizes) == 3:
        for split in TRAIN_VAL_TEST:
            os.makedirs(os.path.join(output_dataset_path, 'images', split), exist_ok=False)
            os.makedirs(os.path.join(output_dataset_path, 'labels', split), exist_ok=False)
    else:
        for split in TRAIN_VAL:
            os.makedirs(os.path.join(output_dataset_path, 'images', split), exist_ok=False)
            os.makedirs(os.path.join(output_dataset_path, 'labels', split), exist_ok=False)
        
    cp_or_mv = shutil.copy if copy_from_source else shutil.move
        
    if len(split_sizes) == 3:
        imgs_rest, imgs_test, txts_rest, txts_test = train_test_split(
            imgs_rest, txts_rest, test_size=split_sizes[2] / sum(split_sizes), random_state=42
        )
        
        for img, txt in zip(imgs_test, txts_test):
            cp_or_mv(img, os.path.join(output_dataset_path, 'images', 'test'))
            cp_or_mv(txt, os.path.join(output_dataset_path, 'labels', 'test'))
            
    imgs_train, imgs_val, txts_train, txts_val = train_test_split(
        imgs_rest, txts_rest, test_size=split_sizes[1] / sum(split_sizes[:2]), random_state=42
    )
    
    for img, txt in zip(imgs_val, txts_val):
        cp_or_mv(img, os.path.join(output_dataset_path, 'images', 'val'))
        cp_or_mv(txt, os.path.join(output_dataset_path, 'labels', 'val'))
        
    for img, txt in zip(imgs_train, txts_train):
        cp_or_mv(img, os.path.join(output_dataset_path, 'images', 'train'))
        cp_or_mv(txt, os.path.join(output_dataset_path, 'labels', 'train'))
        
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
    if not name.endswith('.yaml'):
        name += '.yaml'
    filename = os.path.join(OUTPUT_DIR, name)
    
    dataset_yaml = {
        "path": dataset_path,
        **{sp: f"images/{sp}" for sp in splits},
        "names": {i: cls for i, cls in enumerate(classes)}
    }
    
    with open(filename, 'w') as f:
        yaml.dump(dataset_yaml, f, allow_unicode=True, sort_keys=False)
        
    if copy_to_root:
        shutil.copy(filename, os.path.join(ROOT_DIR, name))
