from src.tools.QuadExplorer import QuAD
from src.tools.project_paths import StressTestPaths, QADatasetPaths


def main():
    mlqa = QuAD(QADatasetPaths.MLQA)
    xquad = QuAD(QADatasetPaths.XQuAD)
    OOD = QuAD(paragraphs=mlqa.paragraphs + xquad.paragraphs)
    OOD.save(StressTestPaths.OOD, version="OOD")


if __name__ == "__main__":
    main()
