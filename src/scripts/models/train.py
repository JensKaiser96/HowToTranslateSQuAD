import sys

from src.qa.dataset import Dataset
from src.qa.qamodel import QAModel
from src.qa.train import train
from src.utils.logging import get_logger

logger = get_logger(__file__, script=True)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Script requires two arguments:"
              f"(1) fuzzy name of the dataset which should be used. i.e one of {Dataset.get_dataset_names()}\n"
              f"(2) fuzzy name of the model it should be saved to, i.e. on of {QAModel.get_model_names()}")
        exit(1)
    fuzzy_dataset = sys.argv[1]
    train_dataset = Dataset.from_fuzzy(fuzzy_dataset)
    fuzzy_model = sys.argv[2]
    save_model = QAModel.from_fuzzy(fuzzy_model)
    train(
        base_model=QAModel.Base,
        train_dataset=train_dataset,
        validation_dataset=Dataset.GermanQUAD.DEV,
        save_path=save_model,
    )
