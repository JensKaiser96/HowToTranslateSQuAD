from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.io.filepaths import PLOTS_PATH
from src.io.utils import str_to_safe_path
from src.qa.dataset import Dataset
from src.qa.evaluate import Evaluation, Result
from src.qa.gelectra import Gelectra


def plot_results(results: Evaluation, name: str, save_path: Path):
    metrics = [
        name for name, _type in Result.__annotations__.items() if _type in (float, int)
    ]
    for metric in metrics:
        data = [getattr(result, metric) for result in results.individual_results]
        # Define the bins
        bins = np.arange(0, 1.1, 0.1)  # Bins from 0 to 1 in 0.1 increments

        # Create the histogram
        plt.figure()
        plt.hist(data, bins=bins, edgecolor="black", alpha=0.7)

        # Add labels and title
        plt.xlabel(metric)
        plt.ylabel("Frequency")
        plt.title(f"{metric} - {name}")

        # Show the plot
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plot_path = str_to_safe_path(save_path / f"hist_{metric}_{name}", ".png")
        plt.savefig(plot_path)


Gelectra.lazy_loading = True

raw_test: Evaluation = Gelectra.RawClean.get_evaluation(Dataset.GermanQUAD.TEST)

model_name = Gelectra.RawClean.name
dataset_name = Dataset.GermanQUAD.TEST.name

plot_results(
    results=raw_test,
    name=f"{model_name} {dataset_name}",
    save_path=Path(PLOTS_PATH) / model_name / dataset_name,
)
