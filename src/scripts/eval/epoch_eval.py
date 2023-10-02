import os

from src.qa.dataset import Dataset
from src.qa.gelectra import Gelectra
from src.utils.logging import get_logger

logger = get_logger(__file__, script=True)

directory_path = "./data/models/gelectra/epoch_test_2023_09_29_12_05_31/checkpoints/"
directory_contents = os.listdir(directory_path)
folders = [
    os.path.join(directory_path, item)
    for item in directory_contents
    if os.path.isdir(os.path.join(directory_path, item))
]

for checkpoint in folders:
    if "/runs" in checkpoint:
        continue
    model = Gelectra(path=checkpoint)
    model.get_evaluation(Dataset.GermanQUAD.TEST)
