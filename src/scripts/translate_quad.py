from src.tools.QuadExplorer import QuAD
from src.tools.project_paths import QADatasetPaths
from src.translate.fairseq import Translator
from src.tools.logging import get_logger

logger = get_logger(__name__)


def main():
    squad1_train = QuAD(QADatasetPaths.SQUAD1_DEV)
    translator = Translator()
    for paragraph in squad1_train.paragraphs:
        paragraph.context = translator.en2de(paragraph.context)
        for qa in paragraph.qas:
            qa.question = translator.en2de(qa.question)
            for answer in qa.answers:
                answer.text = translator.en2de(answer.text)
    squad1_train.save(QADatasetPaths.RAW_SQUAD1_TRAIN, "raw")


if __name__ == "__main__":
    main()
