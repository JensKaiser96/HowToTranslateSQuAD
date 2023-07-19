from collections import Counter

from src.qa.quad import QUAD
from src.utils.logging import get_logger

logger = get_logger(__file__, script=True)


def main():
    squad_raw = QUAD(QUAD.Datasets.Squad1.Translated.Raw)
    answer_counts = Counter()
    for article in squad_raw.data:
        for paragraph in article:
            context = paragraph.context
            for qa in paragraph.qas:
                for answer in qa.answers:
                    count = context.count(answer.text)
                    answer_counts[count] += 1
    logger.info(answer_counts)


if __name__ == '__main__':
    main()
