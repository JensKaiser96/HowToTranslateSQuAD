import random

from src.qa.quad import QUAD
from src.utils.logging import get_logger

logger = get_logger(__file__, script=True)


def create_not(_data: list):
    QUAD(_data=_data).save(
            QUAD.StressTest.Base.NOT, version="GermanQuAD_test_NOT"
        )


def create_dis(_data: list, size: int = 50):
    dis = QUAD(_data=_data)
    dis.data._data = sorted(dis.data, key=lambda p: len(p.context))[:size]
    logger.info(
            f"The longest paragraph has length: {len(dis.data[0].context)}"
        )
    dis.save(QUAD.StressTest.Base.DIS, version="GermanQuAD_test_DIS")


def create_one(_data: list):
    QUAD(_data=_data).save(
            QUAD.StressTest.Base.ONE, version="GermanQuAD_test_ONE"
        )


def split(list_: list, n: int) -> tuple[list]:
    """
    splits the input list `l` into `n` equally (+-1) sized lists
    """
    list_ = list_.copy()
    random.shuffle(list_)
    return (list_[i::n] for i in range(n))


if __name__ == "__main__":
    dataset = QUAD(QUAD.Datasets.GermanQuad.TEST)
    data_NOT, data_DIS, data_ONE = split(dataset.data._data, 3)
    create_not(data_NOT)
    create_dis(data_DIS)
    create_one(data_ONE)
