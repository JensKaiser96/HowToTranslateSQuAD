from collections import Counter

from src.io.filepaths import PLOTS_PATH
from src.nlp_tools.words import question_word_mapping_en_de as mapping
from src.plot import plot_4bars
from src.qa.dataset import Dataset
from src.qa.evaluate_dataset import DatasetEvaluation
from src.utils.logging import get_logger

logger = get_logger(__file__, script=True)


def count_bins(answer_lengths, keys):
    counter = Counter()
    for answer_length in answer_lengths:
        for bin_value in keys:
            if answer_length <= bin_value:
                counter[bin_value] += 1
                break
    return counter


def plot_answer_types(squad, raw, tar, quote):
    keys = list(squad.question_types.keys())

    squad_values = [squad.question_types[key]/squad.number_qa_pairs * 100for key in keys]
    raw_values = [raw.question_types[mapping[key]]/raw.number_qa_pairs * 100 for key in keys]
    tar_values = [tar.question_types[mapping[key]]/tar.number_qa_pairs * 100 for key in keys]
    quote_values = [quote.question_types[mapping[key]]/quote.number_qa_pairs * 100 for key in keys]

    save_path = PLOTS_PATH + "dataset_analysis/question_types"
    plot_4bars(keys, squad_values, raw_values, tar_values, quote_values, save_path)


def plot_answer_lengths(squad, raw, tar, quote):
    keys = [1, 2, 5, 10, 15, 20, 50, 100]

    squad_bins = count_bins(squad.answer_lengths, keys)
    raw_bins = count_bins(raw.answer_lengths, keys)
    tar_bins = count_bins(tar.answer_lengths, keys)
    quote_bins = count_bins(quote.answer_lengths, keys)

    squad_values = [squad_bins[key] / squad.number_qa_pairs * 100 for key in keys]
    raw_values = [raw_bins[key] / raw.number_qa_pairs * 100 for key in keys]
    tar_values = [tar_bins[key]/tar.number_qa_pairs * 100 for key in keys]
    quote_values = [quote_bins[key]/quote.number_qa_pairs * 100 for key in keys]

    save_path = PLOTS_PATH + "dataset_analysis/answer_lengths"

    plot_4bars(keys, squad_values, raw_values, tar_values, quote_values, save_path)


def plot_context_lengths(squad, raw, tar, quote):
    keys = [50, 75, 100, 125, 150, 175, 200, 500]

    squad_bins = count_bins(squad.context_lengths, keys)
    raw_bins = count_bins(raw.context_lengths, keys)
    tar_bins = count_bins(tar.context_lengths, keys)
    quote_bins = count_bins(quote.context_lengths, keys)

    squad_values = [squad_bins[key] / len(squad.context_lengths) * 100 for key in keys]
    raw_values = [raw_bins[key] / len(raw.context_lengths) * 100 for key in keys]
    tar_values = [tar_bins[key] / len(tar.context_lengths) * 100 for key in keys]
    quote_values = [quote_bins[key] / len(quote.context_lengths) * 100 for key in keys]

    save_path = PLOTS_PATH + "dataset_analysis/context_lengths"

    plot_4bars(keys, squad_values, raw_values, tar_values, quote_values, save_path)


def main():
    squad:DatasetEvaluation = Dataset.Squad1.TRAIN.get_evaluation()
    raw:DatasetEvaluation = Dataset.Raw.TRAIN_CLEAN.get_evaluation()
    tar:DatasetEvaluation = Dataset.Tar.TRAIN.get_evaluation()
    quote:DatasetEvaluation = Dataset.Quote.TRAIN.get_evaluation()
    evals = {"squad": squad, "raw": raw, "tar":tar, "quote":quote}
    #plot_answer_types(squad, raw, tar, quote)
    #plot_answer_lengths(squad, raw, tar, quote)
    plot_context_lengths(squad, raw, tar, quote)




"""
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
"""

if __name__ == "__main__":
    main()