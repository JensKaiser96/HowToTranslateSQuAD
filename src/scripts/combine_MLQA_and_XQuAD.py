from src.tools.QUAD import QUAD
from src.tools.project_paths import StressTestPaths, QADatasetPaths


def main():
    mlqa = QUAD(QADatasetPaths.MLQA)
    xquad = QUAD(QADatasetPaths.XQuAD)
    OOD = QUAD(paragraphs=mlqa.paragraphs + xquad.paragraphs)
    OOD.save(StressTestPaths.OOD, version="OOD")


if __name__ == "__main__":
    main()
