import numpy as np

from autocar.ml.training import train_4d_model
from autocar.ml.training import train_6d_model

def test_train_4d_model_returns_working_model(
    tmp_path,
):
    features = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0],
    ] * 10)

    targets = np.ones((40, 1))
    data = np.hstack((features, targets))

    data_path = tmp_path / "training4d.txt"
    np.savetxt(data_path, data)

    model, scaler, history, matrics = train_4d_model(
        data_path,
        epochs=30,
        batch_size=8,
    )

    assert len(history) == 30
    assert np.isfinite(history).all()
    assert history[-1] < history[0]

    scaled_features = scaler.transform(features)
    predictions = model.predict(scaled_features)

    assert predictions.shape == (40, 1)
    assert np.isfinite(predictions).all()

def test_train_6d_model_returns_working_model(
    tmp_path,
):
    features = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [2.0, 3.0, 4.0, 5.0, 6.0],
        [3.0, 4.0, 5.0, 6.0, 7.0],
        [4.0, 5.0, 6.0, 7.0, 8.0],
    ] * 10)

    targets = np.ones((40, 1))
    data = np.hstack((features, targets))

    data_path = tmp_path / "training6d.txt"
    np.savetxt(data_path, data)

    model, scaler, history, metrics = train_6d_model(
        data_path,
        epochs=30,
        batch_size=8,
    )

    assert len(history) == 30
    assert np.isfinite(history).all()
    assert history[-1] < history[0]

    scaled_features = scaler.transform(features)
    predictions = model.predict(scaled_features)

    assert predictions.shape == (40, 1)
    assert np.isfinite(predictions).all()

    assert np.isfinite(metrics["training_mse"])
    assert np.isfinite(metrics["validation_mse"])
