import json
from src.io.filepaths import Datasets, StressTest
from src.utils.io import to_json

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

    @ answer_start.setter
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


class QUAD:
    """
    A Object view on the QuAD structure
    """

    # make dataset paths available trough this class without having to import
    # them explicitly
    Datasets = Datasets
    StressTest = StressTest

    def __init__(self, path: str = "", _data: QuadData = None):
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
        paragraph = Paragraph({QuADKeys.context: context,
                               QuADKeys.qas: [qa._data]})
        self.data._data.append({QuADKeys.paragraphs: [paragraph._data]})

    def save(self, path: str, version: str = ""):
        print(f"saving dataset '{version}' of size: '{len(self.data)}'"
              f"to path: '{path}'")
        if version:
            self.version = version
        to_json(self._data, path)