from autocar.storage.model_io import (save_model, load_model)
from autocar.ml.training import (train_4d_model,train_6d_model) 
from autocar.simulation.simulator import Simulator
from autocar.simulation.vehicle import Vehicle
from autocar.simulation.track import load_track
from autocar.storage.trajectory import(save_4d_trajectory,save_6d_trajectory, load_4d_trajectory, load_6d_trajectory)
from autocar.simulation.replay import replay_4d_trajectory
from autocar.gui.app import AutocarApp
import tkinter as tk
from pathlib import Path

def main() -> None:
    project_root = Path(__file__).resolve().parent
    track_path = project_root / "data"/ "track.txt"
    model_path = project_root / "models" / "mlp_4d.npz" 
    data_path = project_root / "data" / "train4dAll.txt"
    output_path = project_root / "outputs"/ "track4D.txt"
    # loading track data
    track = load_track(track_path)
    # 4D mlp
    model,scaler = load_model(model_path)
    if model is None and scaler is None:
        model, scaler, loss_history, matrics = train_4d_model(data_path)
        save_model(model_path,model, scaler)  
    # create vehicle object 
    vehicle = Vehicle(track.start_position[0], track.start_position[1],track.start_position[2])

    simulator = Simulator(track, vehicle, model, scaler) 
    status = simulator.run(max_steps=1000,)
    if status == "goal":
        save_4d_trajectory(output_path,simulator.records)
    print("status:",status)
    print("total steps:",len(simulator.records))
    print("final x:",vehicle.x)
    print("final y:",vehicle.y)
    print("final heading:",vehicle.heading_degrees)
    print("prepare window")
    root = tk.Tk()
    root.title("Autocar Simulatior")
    root.geometry("900x700")
    #root.resizable(False,False)
    print("window has constructed")
    app = AutocarApp(root_window=root,track=track)
    app.draw_track()
    frames = replay_4d_trajectory(track,load_4d_trajectory(output_path))
    app.start_replay(frames)
    root.mainloop()
    print("window has close down")
    #6Dmlp
#    model_path = project_root / "models" / "mlp_6d.npz" 
#    data_path = project_root / "data" / "train6dAll.txt"
#    output_path = project_root / "outputs"/ "track6D.txt"
#    model,scaler = load_model(model_path)
#    if model is None and scaler is None:
#        model, scaler, loss_history, matrics = train_6d_model(data_path)
#        save_model(model_path,model, scaler)  
#    # create vehicle object 
#    vehicle = Vehicle(track.start_position[0], track.start_position[1],track.start_position[2])
#
#    simulator = Simulator(track, vehicle, model, scaler) 
#    status = simulator.run(max_steps=1000,)
#    if status == "goal":
#        save_6d_trajectory(output_path,simulator.records)
#    print("status:",status)
#    print("total steps:",len(simulator.records))
#    print("final x:",vehicle.x)
#    print("final y:",vehicle.y)
#    print("final heading:",vehicle.heading_degrees)
#

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
