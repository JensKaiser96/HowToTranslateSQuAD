import os

import matplotlib.pyplot as plt

from src.io.filepaths import PREDICTIONS_PATH, PLOTS_PATH
from src.qa.evaluate import Evaluation
from src.utils.logging import get_logger

logger = get_logger(__file__, script=True)

"""
├── checkpoints.checkpoint-105720_GermanQuAD.GermanQuAD_test.json
├── checkpoints.checkpoint-10572_GermanQuAD.GermanQuAD_test.json
├── checkpoints.checkpoint-21144_GermanQuAD.GermanQuAD_test.json
├── checkpoints.checkpoint-31716_GermanQuAD.GermanQuAD_test.json
├── checkpoints.checkpoint-42288_GermanQuAD.GermanQuAD_test.json
├── checkpoints.checkpoint-52860_GermanQuAD.GermanQuAD_test.json
├── checkpoints.checkpoint-63432_GermanQuAD.GermanQuAD_test.json
├── checkpoints.checkpoint-74004_GermanQuAD.GermanQuAD_test.json
├── checkpoints.checkpoint-84576_GermanQuAD.GermanQuAD_test.json
"""

def extract_steps(file_name: str):
    return int(file_name.removeprefix("checkpoints.checkpoint-").removesuffix("_GermanQuAD.GermanQuAD_test.json"))


def main():
    logger.info("Loading Evaluation files...")
    evals = {
        extract_steps(file): Evaluation.load(PREDICTIONS_PATH + file)
        for file in os.listdir(PREDICTIONS_PATH)
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
    plt.title('EM and F1 Scores for Items')
    plt.legend()

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Show the plot
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_PATH + "epoch_eval_f1_em.png")


if __name__ == '__main__':
    main()
