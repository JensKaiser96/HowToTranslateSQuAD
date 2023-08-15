"""
Module containing all paths to important files
should be a yaml, but eh, maybe one day
"""

plotpath = "./data/plots/"
prediction_path = "data/predictions/"

_DATASETS = "./data/datasets/"
_MODELS = "./data/models/"


class Datasets:
    class GermanQuad:
        _dir_path = _DATASETS + "GermanQuAD/"
        TEST = _dir_path + "GermanQuAD_test.json"
        TRAIN = _dir_path + "GermanQuAD_train.json"

    class Squad1:
        _dir_path = _DATASETS + "SQuAD/"
        TRAIN = _dir_path + "train-v1.1.json"
        DEV = _dir_path + "dev-v1.1.json"

        class Translated:
            class Raw:
                _dir_path = _DATASETS + "RAW_SQUAD/"
                TRAIN = _dir_path + "train-v1.0.json"
                TRAIN_CLEAN = _dir_path + "train_clean-v1.0.json"

    class Squad2:
        _dir_path = _DATASETS + "SQuAD/"
        TRAIN = _dir_path + "train-v2.0.json"
        DEV = _dir_path + "dev-v2.0.json"

    class Mlqa:
        _dir_path = _DATASETS + "MLQA/"
        TEST = _dir_path + "test-context-de-question-de"

    class Xquad:
        _dir_path = _DATASETS + "XQuAD/"
        TEST = _dir_path + "xquad.de.json"


class StressTest:
    _dir_path = _DATASETS + "stress_test/"
    OOD = _dir_path + "OOD.json"
    NOT = _dir_path + "NOT.json"
    DIS = _dir_path + "DIS.json"
    ONE = _dir_path + "ONE.json"

    class Base:
        _dir_path = _DATASETS + "stress_test/base/"
        NOT = _dir_path + "NOT.json"
        DIS = _dir_path + "DIS.json"
        ONE = _dir_path + "ONE.json"


class Models:
    class Alignment:
        _dir_path = "/mount/arbeitsdaten31/studenten1/kaiserjs/models/xlm-align-base/"
        config = _dir_path + "config.json"
        model_path = _dir_path
        bpq = _dir_path + "sentencepiece.bpe.model"
        vocab = _dir_path + "fairseq-dict.txt"

    class QA:
        class Gelectra:
            _dir_path = _MODELS + "gelectra/"
            raw_clean = _dir_path + "raw_clean/"
