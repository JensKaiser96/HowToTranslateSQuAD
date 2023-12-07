"""
Module containing all paths to important files
should be a yaml, but eh, maybe one day
"""

PLOTS_PATH = "./data/plots/"
PREDICTIONS_PATH = "data/predictions/"
RESULTS_PATH = "./data/results/"

DATASETS_PATH = "./data/datasets/"
MODELS_PATH = "./data/models/"


class Datasets:
    class GermanQuad:
        _dir_path = DATASETS_PATH + "GermanQuAD/"
        SMALL = _dir_path + "GermanQUAD_small.json"
        TEST = _dir_path + "GermanQuAD_test.json"
        TRAIN = _dir_path + "GermanQuAD_train.json"
        DEV = _dir_path + "GermanQuAD_dev.json"
        TRAIN_WO_DEV = _dir_path + "GermanQuAD_train_wo_dev.json"

    class Squad1:
        _dir_path = DATASETS_PATH + "SQuAD/"
        TRAIN = _dir_path + "train-v1.1.json"
        DEV = _dir_path + "dev-v1.1.json"
        TRAIN_SMALL = _dir_path + "train-v1.1.small.json"

        class Translated:
            class Raw:
                _dir_path = DATASETS_PATH + "RAW_SQUAD/"
                TRAIN = _dir_path + "train-v1.0.json"
                TRAIN_CLEAN = _dir_path + "train_clean-v1.0.json"
                TRAIN_BACK = _dir_path + "train_clean_back-v1.1.json"

            class Tar:
                _dir_path = DATASETS_PATH + "TAR/"
                TRAIN = _dir_path + "train-v1.1.json"
                TRAIN_BACK = _dir_path + "train_back-v1.1.json"

            class Quote:
                _dir_path = DATASETS_PATH + "QUOTE/"
                TRAIN = _dir_path + "train-v1.1.json"
                TRAIN_BACK = _dir_path + "train_back-v1.1.json"

    class Squad2:
        _dir_path = DATASETS_PATH + "SQuAD/"
        TRAIN = _dir_path + "train-v2.0.json"
        DEV = _dir_path + "dev-v2.0.json"

    class Mlqa:
        _dir_path = DATASETS_PATH + "MLQA/"
        TEST = _dir_path + "test-context-de-question-de"

    class Xquad:
        _dir_path = DATASETS_PATH + "XQuAD/"
        TEST = _dir_path + "xquad.de.json"


class StressTest:
    _dir_path = DATASETS_PATH + "stress_test/"
    OOD = _dir_path + "OOD.json"
    NOT = _dir_path + "NOT.json"
    DIS = _dir_path + "DIS.json"
    ONE = _dir_path + "ONE.json"

    class Base:
        _dir_path = DATASETS_PATH + "stress_test/base/"
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
            _dir_path = MODELS_PATH + "gelectra/"
            raw_clean = _dir_path + "raw_clean/"
            raw_clean_1 = _dir_path + "raw_clean_1/"
            raw_clean_2 = _dir_path + "raw_clean_2/"
            raw_clean_3 = _dir_path + "raw_clean_3/"
            raw_clean_4 = _dir_path + "raw_clean_4/"
            raw_back = _dir_path + "raw_back"
            tar = _dir_path + "tar/"
            tar_back = _dir_path + "tar_back"
            quote = _dir_path + "quote/"
            quote_back = _dir_path + "quote_back"
