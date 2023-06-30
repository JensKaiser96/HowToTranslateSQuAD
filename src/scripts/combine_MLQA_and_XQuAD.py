from src.tools.quad import QUAD
from src.tools.logging import get_logger

logger = get_logger(__file__, script=True)


def main():
    mlqa = QUAD(QUAD.Datasets.MLQA)
    xquad = QUAD(QUAD.Datasets.XQuAD)
    OOD = QUAD(_data=mlqa.data._data + xquad.data._data)
    OOD.save(QUAD.StressTest.OOD, version="OOD")


if __name__ == "__main__":
    main()
