from pathlib import Path
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def evaluate_preds(
    groundtruths_path: Path, preds_path: Path, average: str = "binary"
) -> dict:
    """
    Evaluates the performance of a machine learning model by comparing its predictions with ground truth labels.

    Args:
    - groundtruths_path (Path): Path to the ground truth labels file in CSV format.
    - preds_path (Path): Path to the predictions file in CSV format.
    - average (str): The type of averaging to use when calculating precision, recall, and F1-score.

    Returns:
    A dictionary containing the precision, recall, and F1-score of the model's predictions.
    """
    testing_labels = np.loadtxt(
        open(groundtruths_path, "rb"), delimiter=",", skiprows=1
    )
    y_pred = np.loadtxt(open(preds_path, "rb"), delimiter=",", skiprows=1)

    y_test = np.argmax(testing_labels, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    return {
        "precision": precision_score(y_test, y_pred, average=average),
        "recall": recall_score(y_test, y_pred, average=average),
        "f1": f1_score(y_test, y_pred, average=average),
    }
