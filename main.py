from autocar.ml.training import train_4d_model

def main() -> None:
    print("Autocar simulator project initialized.")
    model, scaler, loss_history,matrics = train_4d_model("~/ML_HW_2/data/train4dAll.txt")
    print("initial loss:", loss_history[0])
    print("final loss:", loss_history[-1])
    print("minimum loss:", min(loss_history))
    print("training MSE:", matrics["training_mse"])
    print("validation MSE:", matrics["validation_mse"])
    


if __name__ == "__main__":
    main()
