import os

from src.qa.dataset import Dataset
from src.qa.qamodel import QAModel
from src.utils.logging import get_logger

logger = get_logger(__file__, script=True)

directory_path = "./data/models/gelectra/epoch_test_opt/checkpoints/"
directory_contents = os.listdir(directory_path)
folders = [
    os.path.join(directory_path, item)
    for item in directory_contents
    if os.path.isdir(os.path.join(directory_path, item))
    and "runs" not in item
]

for checkpoint in folders:
    model = QAModel(path=checkpoint)
    model.get_evaluation(Dataset.GermanQUAD.TEST)
