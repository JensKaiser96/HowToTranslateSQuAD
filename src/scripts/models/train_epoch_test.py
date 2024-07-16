from src.io.filepaths import Models, MODELS
from src.qa.dataset import Dataset
from src.qa.qamodel import QAModel
from src.qa.train import train
from src.utils.logging import get_logger

logger = get_logger(__file__, script=True)
train(
    base_model=QAModel(Models.QA.Gelectra.BASE),
    train_dataset=Dataset.Raw.TRAIN_CLEAN,
    validation_dataset=Dataset.GermanQUAD.TEST,
    save_path=MODELS + "gelectra/epoch_test_1e-5/",
    save_strategy="epoch",
    num_train_epochs=10,
    learning_rate=1e-5
)
