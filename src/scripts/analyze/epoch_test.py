import os

import matplotlib.pyplot as plt

from src.io.filepaths import PREDICTIONS_PATH, PLOTS_PATH
from src.qa.evaluate import Evaluation
from src.utils.logging import get_logger

logger = get_logger(__file__, script=True)


def extract_steps(file_name: str):
    return int(file_name.removeprefix("checkpoints.checkpoint-").removesuffix("_GermanQuAD.GermanQuAD_test.json"))


def main():
    logger.info("Loading Evaluation files...")
    lr = "lr1e-5"
    evals = {
        extract_steps(file): Evaluation.load(f"{PREDICTIONS_PATH}epoch_eval_{lr}/{file}")
        for file in os.listdir(PREDICTIONS_PATH + "/epoch_eval_" + lr)
        if file.startswith("checkpoints.")
    }
    logger.info("Extracting Data ...")
    evals = dict(sorted(evals.items()))

    keys = list(evals.keys())
    em_values = [item.EM for item in evals.values()]
    f1_values = [item.F1 for item in evals.values()]

    logger.info("Plotting Data ...")
    # Create a line plot for EM values
    plt.plot(keys, em_values, label='EM', color='blue')

    # Create a line plot for F1 values
    plt.plot(keys, f1_values, label='F1', color='green')

    # Add labels, title, and legend
    plt.xlabel('Steps')
    plt.ylabel('Score')
    plt.title(f'EM/F1 over training at lr: {lr}')
    plt.legend()

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Show the plot
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_PATH}epoch_eval_{lr}.png")


if __name__ == '__main__':
    main()
