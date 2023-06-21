import random
from src.QuadExplorer import QuAD


def create_NOT(dataset: QuAD, size: int = 100):
    pass


def create_DIS(dataset: QuAD, size: int = 50):
    short_paragraphs = [p for p in dataset.paragraphs if len(p) < 750]
    sample = random.sample(short_paragraphs, size)
    dis = QuAD(paragraphs=sample)
    dis.save("./data/datasets/stress_test/DIS.json")


def create_ONE(dataset: QuAD, size: int = 100):
    pass


if __name__ == "__main__":
    dataset = QuAD("./data/datasets/GermanQuAD/GermanQuAD_test.json")
    create_NOT(dataset, 50)
    create_DIS(dataset, 50)
    create_ONE(dataset, 50)
