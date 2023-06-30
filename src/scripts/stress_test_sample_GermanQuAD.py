import random
from src.tools.quad import QUAD
from src.tools.logging import get_logger

logger = get_logger(__file__, script=True)


def create_NOT(_data: list):
    QUAD(_data=_data).save(
            QUAD.StressTest.NOT, version="GermanQuAD_test_NOT")


def create_DIS(_data: list, size: int = 50):
    dis = QUAD(_data=_data)
    dis.data._data = sorted(dis.data, key=lambda p: len(p.context))[:size]
    logger.info(f"The longest paragraph has length: {len(dis.data[0].context)}")
    dis.save(QUAD.StressTest.DIS, version="GermanQuAD_test_DIS")


def create_ONE(_data: list):
    QUAD(_data=_data).save(
            QUAD.StressTest.ONE, version="GermanQuAD_test_ONE")


def split(list_: list, n: int) -> tuple[list]:
    """
    splits the input list `l` into `n` equally (+-1) sized lists
    """
    list_ = list_.copy()
    random.shuffle(list_)
    return (list_[i::n] for i in range(n))


if __name__ == "__main__":
    dataset = QUAD(QUAD.Datasets.GermanQuADTest)
    data_NOT, data_DIS, data_ONE = split(dataset.data._data, 3)
    create_NOT(data_NOT)
    create_DIS(data_DIS)
    create_ONE(data_ONE)
