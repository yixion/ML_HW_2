from pathlib import Path

import numpy as np

from autocar.ml.mlp import MLPRegressor
from autocar.ml.preprocessing import StandardScaler



def save_model(path,model,scaler):
    saved_values = {}
    for index, weight in enumerate(model.W):
        saved_values[f"weight_{index}"]=weight
    saved_values["layer_sizes"] = model.layer_sizes
    saved_values["learning_rate"] = model.l_r
    saved_values["scaler_mean"] = scaler.mean_
    saved_values["scaler_scale"] = scaler.scale_
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    np.savez(path, **saved_values)


def load_model(path):
    file_path = Path(path)
    print("start loading model...")
    if not file_path.is_file():
       return None, None 
    with np.load(file_path,allow_pickle=False) as saved_data:
        scaler = StandardScaler()
        scaler.mean_ = saved_data["scaler_mean"]
        scaler.scale_ = saved_data["scaler_scale"]
        layer_sizes = tuple(saved_data["layer_sizes"].tolist())
        model = MLPRegressor(layer_sizes[0],layer_sizes[1:-1])
        model.l_r = saved_data["learning_rate"]
        for index in range(len(layer_sizes)-1):
            model.W[index] = saved_data[f"weight_{index}"]
    return model, scaler

 
