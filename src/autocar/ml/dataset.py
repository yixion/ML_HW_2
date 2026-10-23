from pathlib import Path
import numpy as np

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
    X,Y = np.split(data,[data.shape[1]-1],axis=1)
    return X,Y

def load_4d_training_data(path):
    return _split_features_and_target(_load_training_dataset(path,4))

def load_6d_training_data(path):
    return _split_features_and_target(_load_training_dataset(path,6))



if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    dataset_path =  PROJECT_ROOT / "data"/ "train4dAll.txt" 
    feature, target = load_4d_training_data(dataset_path)
    print(f"feature shape = {feature.shape}")
    print(f"target shape = {target.shape}")

