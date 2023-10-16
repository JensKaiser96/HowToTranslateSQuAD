from src.io.filepaths import Models
from src.qa.dataset import Dataset
from src.qa.qamodel import QAModel
from src.qa.train import train
from src.utils.logging import get_logger

logger = get_logger(__file__, script=True)


train(
    base_model=QAModel.Base,
    train_dataset=Dataset.Tar.TRAIN,
    validation_dataset=Dataset.GermanQUAD.DEV,
    save_path=Models.QA.Gelectra.tar,
)
