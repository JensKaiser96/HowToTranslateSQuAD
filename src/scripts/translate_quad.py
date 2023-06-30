from src.tools.quad import QUAD
from src.translate.fairseq import Translator
from src.tools.logging import get_logger

logger = get_logger(__file__, script=True)


def main():
    squad1_train = QUAD(QUAD.Datasets.SQUAD1_DEV)  # todo: set to TRAIN instead of DEV
    translator = Translator()
    for paragraph in squad1_train.paragraphs:
        paragraph.context = translator.en2de(paragraph.context)
        for qa in paragraph.qas:
            qa.question = translator.en2de(qa.question)
            for answer in qa.answers:
                answer.text = translator.en2de(answer.text)
    squad1_train.save(QUAD.Datasets.RAW_SQUAD1_TRAIN, "raw")


if __name__ == "__main__":
    main()
