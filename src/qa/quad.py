import json

import datasets

from src.io.filepaths import Datasets, StressTest
from src.io.utils import to_json
from src.qa.train_util import prepare_train_features, flatten_quad
from src.utils.logging import get_logger

logger = get_logger(__name__)

"""
quad structure:
    (version: <str>)
    data:
        [
            (title: <str>)
            paragraphs:
                [
                    (document_id: <str>)
                    context: <str>
                    qas:
                        [
                            question: <str>
                            (is_impossible: <str>)
                            answers: [
                                text: <str>,
                                answer_start: <int>,
                                id: <str>
                                ]
                        ]
                ]

        ]

"""


class QuADKeys:
    data = "data"
    version = "version"
    paragraphs = "paragraphs"
    context = "context"
    qas = "qas"
    question = "question"
    answers = "answers"
    text = "text"
    answer_start = "answer_start"


class Answer:
    def __init__(self, _data: dict):
        self._data = _data

    @property
    def text(self) -> str:
        return self._data[QuADKeys.text]

    @text.setter
    def text(self, _text):
        self._data[QuADKeys.text] = _text

    @property
    def answer_start(self) -> int:
        return int(self._data[QuADKeys.answer_start])

    @answer_start.setter
    def answer_start(self, _answer_start: int):
        self._data[QuADKeys.answer_start] = str(_answer_start)


class Answers:
    def __init__(self, _data):
        self._data = _data

    def __getitem__(self, index) -> Answer:
        return Answer(self._data[index])

    def __len__(self):
        return len(self._data)


class QA:
    def __init__(self, _data: dict):
        self._data = _data

    @property
    def question(self) -> str:
        return self._data[QuADKeys.question]

    @question.setter
    def question(self, _question):
        self._data[QuADKeys.question] = _question

    @property
    def answers(self) -> Answers:
        return Answers(self._data[QuADKeys.answers])


class QAS:
    def __init__(self, _data: list):
        self._data = _data

    def __getitem__(self, index: int) -> QA:
        return QA(self._data[index])

    def __len__(self):
        return len(self._data)


class Paragraph:
    def __init__(self, _data: dict):
        self._data = _data

    @property
    def qas(self) -> QAS:
        return QAS(self._data[QuADKeys.qas])

    @property
    def context(self) -> str:
        return self._data[QuADKeys.context]

    @context.setter
    def context(self, _context):
        self._data[QuADKeys.context] = _context


class Paragraphs:
    def __init__(self, _data: list):
        self._data = _data

    def __getitem__(self, index: int) -> Paragraph:
        return Paragraph(self._data[index])

    def __len__(self):
        return len(self._data)


class QuadData:
    def __init__(self, _data: list):
        self._data = _data

    def __getitem__(self, index: int) -> Paragraphs:
        return Paragraphs(self._data[index][QuADKeys.paragraphs])

    def __len__(self):
        return len(self._data)


# TODO make all datasets accessable via static class
class QUAD:
    """
    A Object view on the QuAD structure
    """

    class Squad1:
        @classmethod
        @property
        def TRAIN(cls):
            return QUAD(Datasets.Squad1.TRAIN)

        @classmethod
        @property
        def DEV(cls):
            return QUAD(Datasets.Squad1.DEV)

    class GermanQUAD:
        @classmethod
        @property
        def TRAIN(cls):
            return QUAD(Datasets.GermanQuad.TRAIN)

        @classmethod
        @property
        def TEST(cls):
            return QUAD(Datasets.GermanQuad.TEST)

    class MLQA:
        @classmethod
        @property
        def TEST(cls):
            return QUAD(Datasets.Mlqa.TEST)

    class XQUAD:
        @classmethod
        @property
        def TEST(self):
            return QUAD(Datasets.Xquad.TEST)

    class Raw:
        @classmethod
        @property
        def TRAIN(cls):
            return QUAD(Datasets.Squad1.Translated.Raw.TRAIN)

        @classmethod
        @property
        def TRAIN_CLEAN(cls):
            return QUAD(Datasets.Squad1.Translated.Raw.TRAIN_CLEAN)

    class StressTest:
        class Base:
            @classmethod
            @property
            def DIS(cls):
                return QUAD(StressTest.Base.DIS)

            @classmethod
            @property
            def NOT(cls):
                return QUAD(StressTest.Base.NOT)

            @classmethod
            @property
            def ONE(cls):
                return QUAD(StressTest.Base.ONE)

        @classmethod
        @property
        def DIS(cls):
            return QUAD(StressTest.DIS)

        @classmethod
        @property
        def NOT(cls):
            return QUAD(StressTest.NOT)

        @classmethod
        @property
        def ONE(cls):
            return QUAD(StressTest.ONE)

        @classmethod
        @property
        def OOD(cls):
            return QUAD(StressTest.OOD)

    def __init__(self, path: str = "", _data: QuadData = None):
        self.path = path
        if path:
            self._data = self._load(path)
        if _data:
            self._data = {QuADKeys.data: _data}
        if not path and not _data:
            self._data = {QuADKeys.data: []}

    @property
    def data(self) -> QuadData:
        return QuadData(self._data[QuADKeys.data])

    @property
    def version(self) -> str:
        return self._data[QuADKeys.version]

    @version.setter
    def version(self, _version: str):
        self._data[QuADKeys.version] = _version

    @staticmethod
    def _load(path: str) -> dict:
        with open(path, mode="r", encoding="utf-8") as f_in:
            return json.load(f_in)

    def add_unanswerable_question(self, context: str, question: str):
        qa = QA({QuADKeys.question: question, QuADKeys.answers: []})
        paragraph = Paragraph({QuADKeys.context: context, QuADKeys.qas: [qa._data]})
        self.data._data.append({QuADKeys.paragraphs: [paragraph._data]})

    def save(self, path: str, version: str = ""):
        logger.info(
            f"saving dataset '{version}' of size: '{len(self.data)} to path: '{path}'"
        )
        if version:
            self.version = version
        to_json(self._data, path)

    def as_hf_dataset(self, tokenizer, split: str = "train"):
        """
        returns the dataset defined at the path as a HuggingFace Dataset. Note this completely ignores the content of
        the QUAD Object, only the data saved to the path is loaded.
        """
        if not self.path:
            raise AttributeError(
                "No path to load from specified. The HuggingFace dataset is loaded directly from the "
                "file, and not the actual QUAD Object"
            )
        raw_dataset = datasets.load_dataset(
            "json", data_files=self.path, field="data", split=split
        )
        flatt_dataset = raw_dataset.map(
            flatten_quad, batched=True, remove_columns=raw_dataset.column_names
        )
        tokenized_dataset = flatt_dataset.map(
            prepare_train_features,
            batched=True,
            remove_columns=flatt_dataset.column_names,
            fn_kwargs={"tokenizer": tokenizer},
        )
        tokenized_dataset.set_format("torch")
        return tokenized_dataset
