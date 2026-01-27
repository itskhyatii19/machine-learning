from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def regression_metrics(y_true, y_pred, verbose=True):
    """
    Compute regression evaluation metrics.

    Returns:
        dict: MAE, MSE, RMSE, R2
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

    if verbose:
        print("\nModel Performance")
        for key, value in metrics.items():
            print(f"{key}: {value}")

    return metrics
