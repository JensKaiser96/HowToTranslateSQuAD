from src.io.filepaths import StressTest, Datasets
from src.qa.dataset import Dataset
from src.utils.logging import get_logger

logger = get_logger(__file__, script=True)


def main():
    mlqa = Dataset.load(Datasets.MLQA.TEST)
    xquad = Dataset.load(Datasets.XQuAD.TEST)
    OOD = Dataset(data=mlqa.data + xquad.data)
    OOD.save(StressTest.OOD, version="OOD")


if __name__ == "__main__":
    main()
