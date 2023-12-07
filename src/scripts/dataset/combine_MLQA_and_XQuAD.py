from src.io.filepaths import StressTest
from src.qa.dataset import Dataset
from src.utils.logging import get_logger

logger = get_logger(__file__, script=True)


def main():
    mlqa: Dataset = Dataset.MLQA.TEST
    xquad: Dataset = Dataset.XQUAD.TEST
    OOD = Dataset(data=mlqa.data + xquad.data)
    OOD.save(StressTest.OOD, version="OOD")


if __name__ == "__main__":
    main()
