"""Actual vs predicted visualization and error metrics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from preprocessing import inverse_transform_close


def metrics_report(y_true_price: np.ndarray, y_pred_price: np.ndarray) -> dict[str, float]:
    return {
        "mse": float(mean_squared_error(y_true_price, y_pred_price)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_price, y_pred_price))),
        "mae": float(mean_absolute_error(y_true_price, y_pred_price)),
    }


def plot_actual_vs_predicted(
    y_true_price: np.ndarray,
    y_pred_price: np.ndarray,
    title: str,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(y_true_price, label="Actual", color="tab:blue", linewidth=1.2)
    plt.plot(y_pred_price, label="Predicted", color="tab:orange", alpha=0.85, linewidth=1.0)
    plt.title(title)
    plt.xlabel("Test step (chronological)")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


def predictions_to_prices(scaler, y_scaled: np.ndarray, pred_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-transform scaled close values to nominal prices."""
    true_p = inverse_transform_close(scaler, y_scaled)
    pred_p = inverse_transform_close(scaler, pred_scaled.flatten())
    return true_p, pred_p
