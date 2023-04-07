import Runner as runner
import DataLoader as data_loader
import evaluate
from pathlib import Path


path_D1 = Path("./datasets-input-neural-network/original/DatasetD1/")
path_D2 = Path("./datasets-input-neural-network/original/DatasetD2/")

data_loader.combine_and_output_experiment_data(path_D1, "groundtruth")
data_loader.combine_and_output_experiment_data(path_D1, "prediction")
data_loader.combine_and_output_experiment_data(path_D2, "groundtruth")
data_loader.combine_and_output_experiment_data(path_D2, "prediction")

print(
    evaluate.evaluate_preds(
        path_D1 / "GroundTruth.csv", path_D1 / "Prediction.csv", "macro"
    )
)
print(evaluate.evaluate_preds(path_D2 / "GroundTruth.csv", path_D2 / "Prediction.csv"))

runner.displayCM(path_D2)
runner.displayCM(path_D1)
