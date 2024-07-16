from src.io.filepaths import Models
from src.qa.dataset import Dataset
from src.qa.qamodel import QAModel
from src.utils.logging import get_logger

logger = get_logger(__file__)


def main():
    raw_clean_model = QAModel(Models.QA.Gelectra.RAW_CLEAN)
    raw_clean_model.get_evaluation(Dataset.GermanQUAD.TEST, redo=True)


if __name__ == "__main__":
    main()
