from src.qa.quad import QUAD
from src.tar.translate import Translator
from src.utils.logging import get_logger
import tqdm

logger = get_logger(__file__, script=True)


def main():
    squad1_train = QUAD(QUAD.Datasets.Squad1.TRAIN)
    translator = Translator()
    for entry in tqdm.tqdm(squad1_train.data):
        for paragraph in tqdm.tqdm(entry):
            paragraph.context = translator.en2de(paragraph.context)
            for qa in paragraph.qas:
                qa.question = translator.en2de(qa.question)
                for answer in qa.answers:
                    answer.text = translator.en2de(answer.text)
    squad1_train.save(QUAD.Datasets.Squad1.Translated.Raw.TRAIN, "raw")


if __name__ == "__main__":
    main()
