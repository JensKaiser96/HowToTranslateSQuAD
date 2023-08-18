import tqdm

from src.io.filepaths import Datasets
from src.qa.quad import QUAD, Paragraphs
from src.tar.translate import Translator
from src.utils.logging import get_logger

logger = get_logger(__file__, script=True)


def main():
    squad1_train = QUAD.Squad1.TRAIN
    translated = QUAD()
    translator = Translator()
    for entry in tqdm.tqdm(squad1_train.data, position=0):
        translated_entry = Paragraphs(entry._data.copy())
        for paragraph in tqdm.tqdm(translated_entry, position=1):
            paragraph.context = translator.en2de(paragraph.context)
            for qa in paragraph.qas:
                qa.question = translator.en2de(qa.question)
                for answer in qa.answers:
                    answer.text = translator.en2de(answer.text)
        translated.data._data.append(translated_entry._data)
        translated.save(Datasets.Squad1.Translated.Raw.TRAIN, "raw")


if __name__ == "__main__":
    main()
