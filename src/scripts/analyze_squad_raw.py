from collections import Counter
import matplotlib.pyplot as plt

from src.qa.quad import QUAD
from src.io.utils import save_plt
from src.utils.logging import get_logger

logger = get_logger(__file__, script=True)


def get_answer_counts():
    squad_raw = QUAD(QUAD.Datasets.Squad1.Translated.Raw.TRAIN)
    answer_counts = Counter()
    for article in squad_raw.data:
        for paragraph in article:
            context = paragraph.context
            for qa in paragraph.qas:
                for answer in qa.answers:
                    count = context.count(answer.text)
                    answer_counts[count] += 1
    logger.info(answer_counts)
    return answer_counts


def plot_counts(answer_counts):
    plot_data = [
            answer_counts[1],
            answer_counts[0],
            sum(answer_counts.values()) - answer_counts[1] - answer_counts[0]
            ]
    plot_labels = ["1", "0", ">1"]
    explode = [0.1, 0, 0]
    fig, ax = plt.subplots()

    ax.pie(plot_data, explode=explode, labels=plot_labels, autopct="%1.1f%%",
           startangle=90)
    fig.tight_layout()
    save_plt(plt, "SQuAD1_raw_translation_answer_count_distribution")


def main():
    answer_counts = {
            0: 936,
            1: 29753,
            2: 2581,
            3: 822,
            4: 301,
            5: 143,
            6: 67,
            7: 59,
            8: 28,
            9: 15,
            10: 11,
            12: 8,
            13: 1,
            17: 1}
    if not answer_counts:
        answer_counts = get_answer_counts()
    plot_counts(answer_counts)


if __name__ == '__main__':
    main()
