from pathlib import Path
import numpy as np
import math

def _load_training_dataset(path, expected_columns):
    path = Path(path).expanduser().resolve()
    data = np.loadtxt(path)
    
    if data.ndim != 2:
        raise ValueError(f"資料必須是二維矩陣，目前維度為 {data.ndim}")
    if data.shape[1] != expected_columns:
        raise ValueError(
                f"expect {expected_columns} columns,"
                f"but exactly get the {data.shape[1]} columns"
                )
    invalid_count = np.sum(~np.isfinite(data))
    if invalid_count > 0:
        raise ValueError(f"there are invalid data such as NaN in the dataset")
    return data

def _split_features_and_target(data):
    feature,target = np.split(data,[data.shape[1]-1],axis=1)
    return feature, target

def load_4d_training_data(path):
    return _split_features_and_target(_load_training_dataset(path,4))

def load_6d_training_data(path):
    return _split_features_and_target(_load_training_dataset(path,6))

def train_validation_split(features,target, validation_ratio=0.2, seed=42):
    if features.shape[0] != target.shape[0]:
        raise ValueError("numbers of data of features and target must be the same!!!")
    if not 0<validation_ratio <1:
        raise ValueError("validation_ratio must be in the interval between 0 and 1")
    indices = np.arange(features.shape[0])
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    test_size = math.ceil(features.shape[0]*validation_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    train_features = features[train_indices]
    train_target = target[train_indices]
    validation_features = features[test_indices]
    validation_target = target[test_indices]
    return (train_features, train_target, validation_features, validation_target)


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    dataset_path =  PROJECT_ROOT / "data"/ "train4dAll.txt" 
    features, target = load_4d_training_data(dataset_path)
    print(f"feature shape = {features.shape}")
    print(f"target shape = {target.shape}")
    train_features, train_target, validation_features, validation_target = train_validation_split(features,target)
    print(train_features.shape)
    print(train_target.shape)
    print(validation_features.shape)
    print(validation_target.shape)


