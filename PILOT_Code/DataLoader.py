# Load CSV
import argparse
from typing import List, Tuple
import numpy as np
import pandas as pd
import csv
from pathlib import Path
import re
import sys
from tqdm import tqdm

from preprocess import NLPPreprocessor, convert_labels_to_binary_str

# load training and testing data from CSV files


def load_data(path, num_of_inputs, num_of_categories):
    """==============read training data=============="""
    raw_data = open(path + "/training_data.csv", "rt")
    tr_d = np.loadtxt(raw_data, delimiter=",")
    training_inputs = [np.reshape(x, (num_of_inputs, 1)) for x in tr_d]
    raw_data = open(path + "/training_labels.csv", "rt")
    tr_l = np.loadtxt(raw_data, delimiter=",")

    # train_labels_flat = train_data.iloc[:,0:1].values
    # train_labels_count = np.unique(tr_l).shape[0]

    training_labels = [vectorization(y, num_of_categories) for y in tr_l]
    training_data = list(zip(training_inputs, training_labels))

    """==============read testing data=============="""
    raw_data = open(path + "/testing_data.csv", "rt")
    te_d = np.loadtxt(raw_data, delimiter=",")
    testing_inputs = [np.reshape(x, (num_of_inputs, 1)) for x in te_d]

    # test_labels = test_data.iloc[:,0:1].values
    # test_labels = dense_to_one_hot(test_labels, train_labels_count)
    # test_labels = test_labels.astype(np.uint8)

    test_data = pd.read_csv(path + "/testing_labels.csv", header=None)
    testing_labels = test_data.iloc[:, 0:1].values
    testing_labels = dense_to_one_hot(testing_labels, num_of_categories)
    testing_labels = testing_labels.astype(np.uint8)

    # raw_data = open(path+'/testing_labels.csv', 'rt')
    # testing_labels = np.loadtxt(raw_data, delimiter=",")
    # testing_labels = dense_to_one_hot(testing_labels, num_of_categories)

    testing_data = testing_inputs
    # testing_data = zip(testing_inputs, te_l)

    return (training_data, testing_data, testing_labels)


def convert_mat_data_to_pilot_format(
    root_mat_data: Path, dest_dir: Path, min_df_threshold: int = 4
) -> None:
    """Converts MAT data to pilot format.

    Args:
        root_mat_data (Path): Path to the root directory containing the data files.

    Returns:
        None

    This function takes a root directory path containing data files in MAT format
    and converts it to pilot format. It creates a directory for the converted data
    in DATASETS_DIR/extensive/Round{project_name} and saves the preprocessed training data and labels,
    and the test data and labels in separate CSV files within this directory. T
    he function uses an NLPPreprocessor object to preprocess the data and
    a get_train_data_and_labels_from_files function to extract the training data and
    labels from the data files.

    Example usage:
        convert_mat_data_to_pilot_format(Path('root_directory_path'))
    """
    extens_set_data_path = dest_dir / "extensive"
    extens_set_data_path.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(root_mat_data.glob("data--*.txt"))

    if len(csv_files) == 0:
        print("No data--*.txt files found in the directory: " + root_mat_data)
        sys.exit(1)

    nlp_preprocessor = NLPPreprocessor(min_df_threshold)

    print("Converting MAT data to pilot format...")

    for test_file_idx, test_filename in tqdm(enumerate(csv_files)):
        print(
            f"Converting train and test data and labels from {test_filename} \
            to pilot format..."
        )

        test_proj_name = get_project_name_from_str(str(test_filename))
        cross_val_fold_dir = extens_set_data_path / f"Round{test_proj_name}"
        cross_val_fold_dir.mkdir(parents=True, exist_ok=True)

        # load and preprocess training data first because preprocessing
        # transformers are fit on training data
        train_data, train_labels = get_train_data_and_labels_from_files(
            root_mat_data, csv_files, test_file_idx, str(test_filename)
        )
        train_data_df = nlp_preprocessor.vectorize_data(train_data, "train")

        print(f"The shape of the training data is {train_data_df.shape}")

        # store preprocessed train data and converted train labels
        train_data_df.to_csv(
            cross_val_fold_dir / "training_data.csv", index=False, header=False
        )

        with open(cross_val_fold_dir / "training_labels.csv", "w") as train_lbl_file:
            csv_writer = csv.writer(train_lbl_file)
            csv_writer.writerows(train_labels)

        # load and preprocess test data and labels
        with open(test_filename, "r") as test_file:
            test_data = test_file.readlines()

        test_data_df = nlp_preprocessor.vectorize_data(test_data, "test")
        print(f"The shape of the test data is {test_data_df.shape}")

        test_label_path = root_mat_data / f"label--{test_proj_name}.txt"
        with open(test_label_path, "r") as label_file:
            test_labels = convert_labels_to_binary_str(label_file.readlines())

        # save test data and labels
        test_data_df.to_csv(
            cross_val_fold_dir / "testing_data.csv", index=False, header=False
        )

        with open(cross_val_fold_dir / "testing_labels.csv", "w") as test_lbl_file:
            csv_writer = csv.writer(test_lbl_file)
            csv_writer.writerows(test_labels)

        print(
            f"Converted {test_proj_name} fold cross-validation train and test \ 
            data to pilot format at {cross_val_fold_dir}"
        )

    print("Converted MAT data to pilot format.")


def get_train_data_and_labels_from_files(
    root_dir: Path, paths: List[Path], test_file_idx: int, test_filename: str
) -> Tuple[List[str], List[str]]:
    """
    Reads training data and labels from files.

    Args:
        root_dir (Path): The root directory containing the label files.
        paths (List[Path]): The list of paths to the training data files.
        test_file_idx (int): The index of the test file in `paths`.
        test_filename (str): The name of the test file.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing two lists:
        - The training data, as a list of strings.
        - The training labels, as a list of binary strings.

    Raises:
        FileNotFoundError: If the label or training files cannot the found.
    """
    train_data = []
    train_labels = []

    for train_file_idx, train_filename in enumerate(paths):
        # skip the test file
        if train_file_idx == test_file_idx:
            continue

        with open(train_filename, "r") as data_file:
            train_data.extend(data_file.readlines())

        train_proj_name = get_project_name_from_str(test_filename)
        train_label_path = root_dir / f"label--{train_proj_name}.txt"
        with open(train_label_path, "r") as label_file:
            train_labels.extend(convert_labels_to_binary_str(label_file.readlines()))

    return train_data, train_labels


def get_project_name_from_str(path: str) -> str:
    """Extract the project name from a string.

    Args:
        path (str): The string to extract the project name from.

    Returns: The project name.
    """
    proj_name = re.search(r"--(.*?)\.txt", path).group(1)
    return proj_name


def get_label_data_path_from_proj_name_and_dir(proj_name: str, data_dir: Path):
    """Get the path to the label data for a project given the project name and
    the data directory.

    Args:
        proj_name (str): The name of the project.
        data_dir (Path): The path to the data directory.

    Returns: The path to the label data for the project.
    """
    return data_dir / f"label--{proj_name}.txt"


def combine_and_output_experiment_data(root: str, type: str) -> None:
    """Combine all experiment round CSV files in a directory into a single CSV file.

    Args:
        root (str): The directory containing the CSV files to combine.
        type (str): The type of data to combine (ground truth or prediction labels).
          Possible values are "groundtruth" and "prediction".

    Returns: None
    """

    root = Path(root)
    if type == "groundtruth":
        csv_files = sorted(root.glob("Round*GroundTruth.csv"))
        label = "GroundTruth"
    elif type == "prediction":
        csv_files = sorted(root.glob("Round*Prediction.csv"))
        label = "Prediction"
    else:
        raise ValueError("Invalid type: " + type)

    # Create an empty list to hold the data from all CSV files
    data = []

    # Loop through all CSV files in the directory
    for filename in csv_files:
        # Open the CSV file and read the data
        with open(filename, "r") as csv_file:
            csv_reader = csv.reader(csv_file)
            data.extend(csv_reader)  # append data to list

    # Write the combined data to a new CSV file
    combined_path = root / f"{label}.csv"
    with open(combined_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(data)


def vectorization(j, num_of_categories):
    e = np.zeros((num_of_categories, 1))
    e[int(j)] = 1.0
    return e


# Convert class labels from scalars to one-hot vectors
# 0 => [1 0 0 0 0 0 0 0 0 0]
# 1 => [0 1 0 0 0 0 0 0 0 0]
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load MAT data, preprocess and convert it to a format suitable for PILOT model training and \
        save the preprocessed data to disk as folds for cross-validation."
    )

    parser.add_argument(
        "-p",
        "--datapath",
        type=str,
        required=True,
        help="Root directory containing the MAT experiment data",
    )

    parser.add_argument(
        "-d",
        "--destination",
        type=str,
        required=True,
        help="Root directory to store the preprocessed MAT data \
              (in cross-validation folds) in. Follows the same structure as the PILOT \
                data directory",
    )

    parser.add_argument(
        "-f",
        "--docfreq",
        type=int,
        required=False,
        default=4,
        help="The minimum document frequency for a word to be included in the vocabulary. \
            Words with a lower document frequency will be filtered out.",
    )

    args = parser.parse_args()

    convert_mat_data_to_pilot_format(
        Path(args.datapath), Path(args.destination), args.docfreq
    )
