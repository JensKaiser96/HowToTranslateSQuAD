from src.io.filepaths import Models
from src.qa.dataset import Dataset
from src.qa.gelectra import Gelectra
from src.qa.train import train
from src.utils.logging import get_logger

logger = get_logger(__file__, script=True)


train(
    base_model=Gelectra.Base,
    train_dataset=Dataset.Raw.TRAIN_CLEAN,
    validation_dataset=Dataset.GermanQUAD.TEST,
    save_path=Models.QA.Gelectra.raw_clean_3
)
