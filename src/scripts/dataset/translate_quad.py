import tqdm

from src.io.filepaths import Datasets
from src.qa.dataset import Dataset
from src.tar.translate import Translator
from src.utils.logging import get_logger

logger = get_logger(__file__, script=True)


def main():
    squad1_train: Dataset = Dataset.Squad1.TRAIN
    translated = Dataset(data=[])
    translator = Translator()
    for article in tqdm.tqdm(squad1_train.data, position=0):
        for paragraph in tqdm.tqdm(article, position=1):
            paragraph.context = translator.en2de(paragraph.context)
            for qa in paragraph.qas:
                qa.question = translator.en2de(qa.question)
                for answer in qa.answers:
                    answer.text = translator.en2de(answer.text)
    translated.save(Datasets.Squad1.Translated.Raw.TRAIN, "raw")


if __name__ == "__main__":
    main()
