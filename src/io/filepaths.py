"""
Module containing all paths to important files
"""

_DATASETS = "./data/datasets/"

class Dataset:
    class GermanQuad:
        _dir_path = _DATASETS + "GermanQuAD"
        Test = _dir_path + "GermanQuAD_test.json"
        Train = _dir_path + "GermanQuAD_train.json"
    class Squad1:
        _dir_path
    SQUAD1_TRAIN = _path_prefix + "/SQuAD/train-v1.1.json"
    SQUAD2_TRAIN = _path_prefix + "./data/datasets/SQuAD/train-v2.0.json"
    SQUAD1_DEV = _path_prefix + "./data/datasets/SQuAD/dev-v1.1.json"
    SQUAD2_DEV = _path_prefix + "./data/datasets/SQuAD/dev-v2.0.json"
    MLQA = _path_prefix + "./data/datasets/MLQA/test-context-de-question-de.json"
    XQuAD = _path_prefix + "./data/datasets/XQuAD/xquad.de.json"
    RAW_SQUAD1_TRAIN = _path_prefix + "./data/datasets/RAW_SQUAD/train-v1.0.json"

class StressTest:
    OOD = "./data/datasets/stress_test/OOD.json"
    NOT = "./data/datasets/stress_test/NOT.json"
    DIS = "./data/datasets/stress_test/DIS.json"
    ONE = "./data/datasets/stress_test/ONE.json"

    class Base:
        NOT = "./data/datasets/stress_test/base/NOT.json"
        DIS = "./data/datasets/stress_test/base/DIS.json"
        ONE = "./data/datasets/stress_test/base/ONE.json"


class StressTestPaths:
    OOD = "./data/datasets/stress_test/OOD.json"
    NOT = "./data/datasets/stress_test/NOT.json"
    DIS = "./data/datasets/stress_test/DIS.json"
    ONE = "./data/datasets/stress_test/ONE.json"

class Alignment:
    config = ""
    model = ""
    bpq = ""
    tokenizer = ""
