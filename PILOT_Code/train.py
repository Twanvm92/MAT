import argparse
import Runner as runner


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model with specified arguments."
    )

    parser.add_argument(
        "-p",
        "--datapath",
        type=str,
        required=True,
        help="Root path of the dataset to be used for model training.",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=100,
        help="Amount of epochs used for training. Default is 100.",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=3.0,
        help="Learning rate used for training the model. Default is 3.0.",
    )
    parser.add_argument(
        "-a",
        "--accuracy",
        type=float,
        default=0.75,
        help="Expected accuracy. Default is 0.75.",
    )
    parser.add_argument(
        "-i",
        "--inputsize",
        type=int,
        required=True,
        help="Input size of the model which is the size of the feature vector.",
    )
    parser.add_argument(
        "-o",
        "--outputsize",
        type=int,
        required=True,
        help="Output size of the model which represents the amount of classes we are trying to predict.",
    )
    parser.add_argument(
        "hidden_layers",
        type=int,
        nargs="+",
        help="Amount of neurons used in a hidden layer.",
    )
    parser.add_argument(
        "-s",
        "--setup",
        choices=["original", "extensive"],
        default="original",
        help='Experimental setup to use. Options are "original" (default) or "extensive".',
    )
    parser.add_argument(
        "-c",
        "--config",
        choices=[1, 2, 3],
        default=1,
        help="Neural network layer configuration to use for PILOT. Only applicable if dataset D1 is used for evaluation. \
              Options are 1 (default), 2 and 3.",
    )

    args = parser.parse_args()

    if args.outputsize < 2:
        print(
            "Output size must be at least 2 to be able to run binary classification or multi-class classification."
        )
        exit(1)

    network_layers = [args.inputsize] + args.hidden_layers + [args.outputsize]

    if args.setup == "original":
        print("Running original setup.")
        if args.outputsize == 2:
            print("Running binary classification.")
        else:
            print(f"Running multi-class classification with {args.outputsize} classes.")

        runner.runExperiment(
            args.datapath,
            network_layers,
            args.epochs,
            args.learning_rate,
            args.accuracy,
            args.config,
        )
    else:
        print("Not implemented yet.")

# TODO in runner still create separate files for each round but also create a file for all rounds (prediction.csv and groundtruth.csv)
# TODO create test.py to create a confusion matrix and ROC curve and caculate accuracy, precision, recall, f1-score, etc.
# TODO add extensive setup
