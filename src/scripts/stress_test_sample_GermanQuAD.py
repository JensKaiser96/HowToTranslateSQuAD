import random
from src.tools.QuadExplorer import QuAD


def create_NOT(paragraphs: list, size: int = 100):
    path = "./data/datasets/stress_test/NOT.json"
    QuAD(paragraphs=paragraphs[:size]).save(path, version="GermanQuAD_test_NOT")


def create_DIS(paragraphs: list, size: int = 50):
    short_paragraphs = sorted(paragraphs, key=lambda p: len(p.context))[:size]
    print(f"The longest paragraph has length: {len(short_paragraphs[-1].context)}")
    path = "./data/datasets/stress_test/DIS.json"
    QuAD(paragraphs=short_paragraphs).save(path, version="GermanQuAD_test_DIS")


def create_ONE(paragraphs: list, size: int = 100):
    path = "./data/datasets/stress_test/ONE.json"
    QuAD(paragraphs=paragraphs[:size]).save(path, version="GermanQuAD_test_ONE")


def split(list_: list, n: int) -> tuple[list]:
    """
    splits the input list `l` into `n` equally (+-1) sized lists
    """
    list_ = list_.copy()
    random.shuffle(list_)
    return (list_[i::n] for i in range(n))


if __name__ == "__main__":
    dataset = QuAD("./data/datasets/GermanQuAD/GermanQuAD_test.json")
    paragraphs_NOT, paragraphs_DIS, paragraphs_ONE = split(dataset.paragraphs, 3)
    create_NOT(paragraphs_NOT)
    create_DIS(paragraphs_DIS)
    create_ONE(paragraphs_ONE)
