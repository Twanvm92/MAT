from pathlib import Path
from typing import Dict, List
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def get_classification_report(
    nn_config: int = 1, y_test: List = None, y_pred: List = None, root: str = None
) -> Dict:
    if root is not None:
        testing_labels = np.loadtxt(open(root / "GroundTruth.csv", "rb"), delimiter=",")
        y_pred = np.loadtxt(
            open(root / f"PredictionC{nn_config}.csv", "rb"), delimiter=","
        )
        y_test = np.argmax(testing_labels, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    elif y_test is not None and y_pred is not None:
        y_test = np.argmax(y_test, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        raise ValueError("Provide either y_test and y_pred or root")

    results = classification_report(y_test, y_pred)

    return results


def displayCM(root: Path, nn_config: int = 1):
    testing_labels = np.loadtxt(open(root / "GroundTruth.csv", "rb"), delimiter=",")
    y_pred = np.loadtxt(open(root / f"PredictionC{nn_config}.csv", "rb"), delimiter=",")

    y_test = np.argmax(testing_labels, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    # ==================================Display precision/recall================================
    # print(classification_report( np.argmax(testing_labels, axis=1), np.argmax(y_pred, axis=1)))

    # ==================================Display confusion matrix================================
    cm = confusion_matrix(y_test, y_pred)

    classes = np.unique(y_test)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap=plt.cm.Blues, cbar=False)
    ax.set(
        xlabel="Prediction",
        ylabel="Real",
        xticklabels=classes,
        yticklabels=classes,
        title="Confusion matrix",
    )
    plt.yticks(rotation=0)

    # fig, ax = plt.subplots(nrows=1, ncols=2)
    plt.show()


def displayROC(root: Path, num_of_categories: int, nn_config: int = 1):
    testing_labels = np.loadtxt(open(root / "GroundTruth.csv", "rb"), delimiter=",")
    y_pred = np.loadtxt(open(root / f"PredictionC{nn_config}.csv", "rb"), delimiter=",")
    # ==================================Display precision/recall================================
    print(get_classification_report(y_test=testing_labels, y_pred=y_pred))

    # ==================================Display confusion matrix================================
    cm = confusion_matrix(np.argmax(testing_labels, axis=1), np.argmax(y_pred, axis=1))
    y_test = np.argmax(testing_labels, axis=1)
    classes = np.unique(y_test)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap=plt.cm.Blues, cbar=False)
    ax.set(
        xlabel="Pred",
        ylabel="True",
        xticklabels=classes,
        yticklabels=classes,
        title="Confusion matrix",
    )
    plt.yticks(rotation=0)

    fig, ax = plt.subplots(nrows=1, ncols=2)
    plt.show()
    # ==================================Display the ROC curvers==================================
    n_classes = num_of_categories

    from scipy import interp

    # import matplotlib.pyplot as plt
    from itertools import cycle
    from sklearn.metrics import roc_curve, auc

    # Plot linewidth.
    lw = 2

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(testing_labels[:, i], y_pred[:, i])
        # print(testing_labels[:, i])
        # print(y_score[:, i])

        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(testing_labels.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # cf_matrix = confusion_matrix(testing_labels.ravel(), y_pred.ravel())
    # print(cf_matrix)

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this point
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(1)

    colors = cycle(
        [
            "blue",
            "gray",
            "silver",
            "aqua",
            "darkorange",
            "cornflowerblue",
            "olivedrab",
            "deepskyblue",
            "slategray",
        ]
    )
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="Class {0} (AUC = {1:0.2f})" "".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC")  # to multi-class
    plt.legend(loc="lower right")
    plt.show()
