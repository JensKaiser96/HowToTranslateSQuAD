from src.tools.QuadExplorer import QuAD


def main():
    mlqa = QuAD(QuAD.Datasets.MLQA)
    xquad = QuAD(QuAD.Datasets.XQuAD)
    OOD = QuAD(paragraphs=mlqa.paragraphs + xquad.paragraphs)
    OOD.save("./data/datasets/stress_test/OOD.json", version="OOD")


if __name__ == "__main__":
    main()
