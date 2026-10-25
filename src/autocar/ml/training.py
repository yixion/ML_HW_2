from autocar.ml.dataset import (load_4d_training_data,
                    load_6d_training_data,
                    train_validation_split)
from autocar.ml.preprocessing import StandardScaler
from autocar.ml.mlp import (MLPRegressor,mean_squared_error)

def train_4d_model(data_path, epochs=500, batch_size=32):
    features, targets = load_4d_training_data(data_path)
    training_features, training_targets, validation_features, validation_targets=train_validation_split(features,targets)
    scaler = StandardScaler()
    scaler.fit(training_features)
    scaled_training_features = scaler.transform(training_features)
    scaled_validation_features = scaler.transform(validation_features)
    model = MLPRegressor(
            input_size=3,
            hidden_sizes=(8,8),
            learning_rate=0.01)
    loss_history = model.fit(scaled_training_features, training_targets, epochs, batch_size)
    training_predictions = model.predict(scaled_training_features)
    validation_predictions = model.predict(scaled_validation_features)
    training_MSE = mean_squared_error(training_targets, training_predictions) 
    validation_MSE = mean_squared_error(validation_targets, validation_predictions) 
    # output result
    print(f'initial training loss = {loss_history[0]}')
    print(f'final training loss = {loss_history[-1]}')
    print(f'training MSE = {training_MSE}')
    print(f'validation MSE = {validation_MSE}')
    return model, scaler, loss_history


