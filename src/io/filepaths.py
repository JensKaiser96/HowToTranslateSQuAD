from pathlib import Path
"""
Module containing all paths to important files
should be a yaml, but eh, maybe one day
"""


class Paths:
    PROJECT = Path(".")

    DATA = PROJECT / "data"
    PLOTS = DATA / "plots"
    PREDICTIONS = DATA / "predictions"
    RESULTS = DATA / "results"

    DATASETS = DATA / "datasets"
    MODELS = DATA / "models"


class Datasets:
    class GermanQuad:
        _dir_path = Paths.DATASETS / "GermanQuAD"
        SMALL = _dir_path / "GermanQUAD_small.json"
        TEST = _dir_path / "GermanQuAD_test.json"
        TRAIN = _dir_path / "GermanQuAD_train.json"
        DEV = _dir_path / "GermanQuAD_dev.json"
        TRAIN_WO_DEV = _dir_path / "GermanQuAD_train_wo_dev.json"

    class SQuAD:
        _dir_path = Paths.DATASETS / "SQuAD"
        TRAIN = _dir_path / "train-v1.1.json"
        DEV = _dir_path / "dev-v1.1.json"
        TRAIN_SMALL = _dir_path / "train-v1.1.small.json"

        class Translated:
            class Raw:
                _dir_path = Paths.DATASETS / "RAW_SQUAD"
                TRAIN = _dir_path / "train-v1.0.json"
                TRAIN_CLEAN = _dir_path / "train_clean-v1.0.json"
                TRAIN_BACK = _dir_path / "train_clean_back-v1.1.json"

            class Tar:
                _dir_path = Paths.DATASETS / "TAR"
                TRAIN = _dir_path / "train-v1.1.json"
                TRAIN_BACK = _dir_path / "train_back-v1.1.json"

            class Quote:
                _dir_path = Paths.DATASETS / "QUOTE"
                TRAIN = _dir_path / "train-v1.1.json"
                TRAIN_BACK = _dir_path / "train_back-v1.1.json"

    class SQuAD2:
        _dir_path = Paths.DATASETS / "SQuAD"
        TRAIN = _dir_path / "train-v2.0.json"
        DEV = _dir_path / "dev-v2.0.json"

    class MLQA:
        _dir_path = Paths.DATASETS / "MLQA"
        TEST = _dir_path / "test-context-de-question-de"

    class XQuAD:
        _dir_path = Paths.DATASETS / "XQuAD/"
        TEST = _dir_path / "xquad.de.json"


class StressTest:
    _dir_path = Paths.DATASETS / "stress_test"
    OOD = _dir_path / "OOD.json"
    NOT = _dir_path / "NOT.json"
    DIS = _dir_path / "DIS.json"
    ONE = _dir_path / "ONE.json"

    class Base:
        _dir_path = Paths.DATASETS / "stress_test/base"
        NOT = _dir_path / "NOT.json"
        DIS = _dir_path / "DIS.json"
        ONE = _dir_path / "ONE.json"


class Models:
    class Alignment:
        _dir_path = Paths.MODELS / "xlm-align-base"
        CONFIG = _dir_path / "config.json"
        MODEL_PATH = _dir_path
        BPQ = _dir_path / "sentencepiece.bpe.model"
        VOCAB = _dir_path / "fairseq-dict.txt"

    class QA:
        class Gelectra:
            _dir_path = Paths.MODELS / "gelectra"
            RAW_CLEAN = _dir_path / "raw_clean"
            RAW_CLEAN_1 = _dir_path / "raw_clean_1"
            RAW_CLEAN_2 = _dir_path / "raw_clean_2"
            RAW_CLEAN_3 = _dir_path / "raw_clean_3"
            RAW_CLEAN_4 = _dir_path / "raw_clean_4"
            RAW_BACK = _dir_path / "raw_back"
            TAR = _dir_path / "tar"
            TAR_BACK = _dir_path / "tar_back"
            QUOTE = _dir_path / "quote"
            QUOTE_BACK = _dir_path / "quote_back"
            # not really a filepath (is a huggingface path) but works anyways
            BASE = "deepset/gelectra-large"
            GERMAN_QUAD = "deepset/gelectra-large-germanquad"

        class Distilbert:
            # again a huggingface path
            ENGLISH_QA = "distilbert-base-cased-distilled-squad"
