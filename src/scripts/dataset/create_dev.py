import random

from src.io.filepaths import Datasets
from src.qa.dataset import Dataset, Article
from src.utils.logging import get_logger

logger = get_logger(__file__, script=True)


def main():
    german_quad_test = Dataset.load(Datasets.GermanQuad.TEST)
    german_quad_train = Dataset.load(Datasets.GermanQuad.TRAIN)
    german_quad_dev = Dataset(data=[])
    german_quad_train_wo_dev = Dataset(data=[])

    # Count QA-Pairs in test
    qas_in_test = sum([len(p.qas) for a in german_quad_test.data for p in a.paragraphs])
    qas_in_train = sum([len(p.qas) for a in german_quad_train.data for p in a.paragraphs])

    current_no_of_qas = 0
    qas_seen = 0

    for article in german_quad_train.data:
        for paragraph in article.paragraphs:
            # threshold, how likely to keep it, UP: Needed, skipped, DOWN: taken, available
            threshold = (qas_in_test-current_no_of_qas) / (qas_in_train-qas_seen)
            qas_seen += len(paragraph.qas)

            if random.random() <= threshold:
                german_quad_dev.data.append(Article(paragraphs=[paragraph]))
                current_no_of_qas += len(paragraph.qas)
            else:
                german_quad_train_wo_dev.data.append(Article(paragraphs=[paragraph]))

    german_quad_dev.save(Datasets.GermanQuad.DEV, "dev")
    german_quad_train_wo_dev.save(Datasets.GermanQuad.TRAIN_WO_DEV, "Train without dev")


if __name__ == "__main__":
    main()
