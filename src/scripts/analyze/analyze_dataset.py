import sys

import matplotlib.pyplot as plt
import numpy as np

from src.io.filepaths import PLOTS_PATH
from src.math.arithmetic import linear_interpolate_zeros
from src.utils.logging import get_logger

logger = get_logger(__file__, script=True)

    logger.info(f"{question_type_counter=}")
    logger.info(f"{answer_type_counter=}")

    logbins = np.logspace(0, 2, 100)

    hist, bins, _ = plt.hist(answer_lengths, bins=logbins)

    bin_centers = (bins[1:] + bins[:-1]) / 2
    curve = np.interp(bin_centers, bin_centers, linear_interpolate_zeros(hist))
    plt.plot(bin_centers, curve)

    plt.xlabel("# tokens in answer")
    plt.ylabel("Frequency")
    plt.xscale("log")
    plt.yscale("log")
    plt.title(f"Answer length distribution in {dataset.name}")
    plt.savefig(f"{PLOTS_PATH}dataset_analysis/answer_lengths{dataset.name}.png", dpi=300)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        fuzzy_name = sys.argv[1]
        main()
    else:
        raise ValueError("Please specify a dataset to analyse.")
