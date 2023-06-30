from src.tools.quad import QUAD
from src.translate.fairseq import Translator
from src.tools.logging import get_logger
import tqdm

logger = get_logger(__file__, script=True)


def main():
    squad1_train = QUAD(QUAD.Datasets.SQUAD1_TRAIN)
    translator = Translator()
    for entry in tqdm.tqdm(squad1_train.data):
        entry.context = translator.en2de(entry.context)
        for qa in entry.qas:
            qa.question = translator.en2de(qa.question)
            for answer in qa.answers:
                answer.text = translator.en2de(answer.text)
    squad1_train.save(QUAD.Datasets.RAW_SQUAD1_TRAIN, "raw")


if __name__ == "__main__":
    main()
