from autocar.ml.training import (train_4d_model,train_6d_model)
from autocar.mlp.mlp import MLPRegressor
from autocar.mlp.preprocessing import StandardScaler
from autocar.simulation.sensors import measure_sensor_distances
from autocar.simulation.vehicle import Vehicle
from autocar.simulation.simulator import Simulator
from autocar.storage.model_io import save_model
from pathlib import Path
import numpy as np

def main() -> None:
    s = Simulator()
    
#    project_root = Path(__file__).resolve().parent
#    print("Autocar simulator project initialized.")
#    data_path = project_root / "data" / "train4dAll.txt"
#    model, scaler, loss_history,matrics = train_4d_model(data_path)
#    print("initial loss:", loss_history[0])
#    print("final loss:", loss_history[-1])
#    print("minimum loss:", min(loss_history))
#    print("training MSE:", matrics["training_mse"])
#    print("validation MSE:", matrics["validation_mse"])
#    model_path = project_root / "models"/"mlp_4d.npz"
#    save_model(model_path,model,scaler)
#        
#    print("==============another data===================") 
#    data_path = project_root / "data" / "train6dAll.txt"
#    model, scaler, loss_history,matrics = train_6d_model(data_path)
#    print("initial loss:", loss_history[0])
#    print("final loss:", loss_history[-1])
#    print("minimum loss:", min(loss_history))
#    print("training MSE:", matrics["training_mse"])
#    print("validation MSE:", matrics["validation_mse"])
#    model_path = project_root / "models"/"mlp_6d.npz"
#    save_model(model_path,model,scaler)
if __name__ == "__main__":
    main()
