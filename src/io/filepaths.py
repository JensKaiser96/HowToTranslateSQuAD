class QADatasetPaths:
    GermanQuADTest = "./data/datasets/GermanQuAD/GermanQuAD_test.json"
    GermanQuADTrain = "./data/datasets/GermanQuAD/GermanQuAD_train.json"
    SQUAD1_TRAIN = "./data/datasets/SQuAD/train-v1.1.json"
    SQUAD2_TRAIN = "./data/datasets/SQuAD/train-v2.0.json"
    SQUAD1_DEV = "./data/datasets/SQuAD/dev-v1.1.json"
    SQUAD2_DEV = "./data/datasets/SQuAD/dev-v2.0.json"
    MLQA = "./data/datasets/MLQA/test-context-de-question-de.json"
    XQuAD = "./data/datasets/XQuAD/xquad.de.json"
    RAW_SQUAD1_TRAIN = "./data/datasets/RAW_SQUAD/train-v1.0.json"


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
