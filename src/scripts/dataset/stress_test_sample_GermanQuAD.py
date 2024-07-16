import random

from src.io.filepaths import StressTest, Datasets
from src.qa.dataset import Dataset
from src.utils.logging import get_logger

logger = get_logger(__file__, script=True)


def create_not(data: list):
    Dataset(data=data).save(StressTest.Base.NOT, version="GermanQuAD_test_NOT")


def create_dis(data: list, size: int = 50):
    dis = Dataset(data=data)
    dis.data._data = sorted(dis.data, key=lambda p: len(p.context))[:size]
    logger.info(f"The longest paragraph has length: {len(dis.data[0].context)}")
    dis.save(StressTest.Base.DIS, version="GermanQuAD_test_DIS")


def create_one(data: list):
    Dataset(data=data).save(StressTest.Base.ONE, version="GermanQuAD_test_ONE")


def split(data: list, n: int) -> tuple[list]:
    """
    splits the input list `l` into `n` equally (+-1) sized lists
    """
    data = data.copy()
    random.shuffle(data)
    return (data[i::n] for i in range(n))


if __name__ == "__main__":
    dataset = Dataset.load(Datasets.GermanQuad.TEST)
    data_NOT, data_DIS, data_ONE = split(dataset.data._data, 3)
    create_not(data_NOT)
    create_dis(data_DIS)
    create_one(data_ONE)
