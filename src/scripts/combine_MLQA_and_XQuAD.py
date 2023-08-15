from src.io.filepaths import StressTest
from src.qa.quad import QUAD
from src.utils.logging import get_logger

logger = get_logger(__file__, script=True)


def main():
    mlqa = QUAD.MLQA.TEST
    xquad = QUAD.XQUAD.TEST
    OOD = QUAD(_data=mlqa.data._data + xquad.data._data)
    OOD.save(StressTest.OOD, version="OOD")


if __name__ == "__main__":
    main()
