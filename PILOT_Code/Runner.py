import DataLoader
import NeuralNetwork as network
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def run(
    path, network_layers, num_of_epochs, learning_rate, expected_accuracy, nn_config
):
    num_of_inputs = network_layers[0]
    num_of_categories = network_layers[len(network_layers) - 1]
    training_data, test_data, testing_labels = DataLoader.load_data(
        path, num_of_inputs, num_of_categories
    )
    num_of_testing_items = len(test_data)
    net = network.Network(path, expected_accuracy, network_layers)
    y_pred = net.SGD(
        training_data, num_of_epochs, 10, learning_rate, test_data=test_data
    )
    y_pred = np.reshape(y_pred, (num_of_testing_items, num_of_categories))

    mat = np.matrix(testing_labels)
    dataframe = pd.DataFrame(data=mat.astype(float))
    dataframe.to_csv(
        path + "GroundTruth.csv",
        mode="w+",
        sep=",",
        header=False,
        float_format="%.5f",
        index=False,
    )

    mat = np.matrix(y_pred)
    dataframe = pd.DataFrame(data=mat.astype(float))
    dataframe.to_csv(
        path + f"PredictionC{nn_config}.csv",
        mode="w+",
        sep=",",
        header=False,
        float_format="%.5f",
        index=False,
    )


def displayCM(root):
    testing_labels = np.loadtxt(
        open(root / "GroundTruth.csv", "rb"), delimiter=",", skiprows=1
    )
    y_pred = np.loadtxt(open(root / "Prediction.csv", "rb"), delimiter=",", skiprows=1)

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


def displayROC(root, num_of_categories):
    testing_labels = np.loadtxt(
        open(root / "GroundTruth.csv", "rb"), delimiter=",", skiprows=1
    )
    y_pred = np.loadtxt(open(root / "Prediction.csv", "rb"), delimiter=",", skiprows=1)

    # ==================================Display precision/recall================================
    print(
        classification_report(
            np.argmax(testing_labels, axis=1), np.argmax(y_pred, axis=1)
        )
    )

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


def runExperiment(
    root, network_layers, num_of_epochs, learning_rate, expected_accuracy, nn_config
):
    print("Experiments started")
    print("Network layers: " + str(network_layers))
    print("Number of epochs: " + str(num_of_epochs))
    print("Learning rate: " + str(learning_rate))
    print("Expected accuracy: " + str(expected_accuracy))
    print("Neural network configuration: C" + str(nn_config))

    # reset the files
    f = open(root + "GroundTruth.csv", "w+")
    f.close()
    f = open(root + "Prediction.csv", "w+")
    f.close()

    for i in range(10):
        print("Round " + str(i + 1))
        path = root + "Round" + str(i + 1)

        # reset each round results
        f = open(path + "Prediction.csv", "w+")
        f.close()

        print(path)
        run(
            path,
            network_layers,
            num_of_epochs,
            learning_rate,
            expected_accuracy,
            nn_config,
        )
