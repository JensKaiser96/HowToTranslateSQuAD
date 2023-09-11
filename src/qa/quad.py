import warnings

from src.utils.logging import get_logger

logger = get_logger(__name__)
logger.warning("DEPRECIATED - use Dataset instead of QUAD")
warnings.warn("QUAD is deprecated, use Dataset instead", DeprecationWarning, 2)
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
    id = "id"
    answers = "answers"
    text = "text"
    answer_start = "answer_start"


class Answer:
    def __init__(self, _data: dict):
        if _data:
            self._data = _data
        else:
            self._data = {
                QuADKeys.text: "",
                QuADKeys.answer_start: 0,
            }

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
        if _data:
            self._data = _data
        else:
            self._data = [Answer()._data]

    def __getitem__(self, index) -> Answer:
        return Answer(self._data[index])

    def __len__(self):
        return len(self._data)


class QA:
    def __init__(self, _data: dict):
        if _data:
            self._data = _data
        else:
            self._data = {
                QuADKeys.question: "",
                QuADKeys.answers: Answers()._data,
                QuADKeys.id: 0,
            }

    @property
    def question(self) -> str:
        return self._data[QuADKeys.question]

    @question.setter
    def question(self, _question):
        self._data[QuADKeys.question] = _question

    @property
    def answers(self) -> Answers:
        return Answers(self._data[QuADKeys.answers])

    @property
    def id(self) -> int:
        if QuADKeys.id in self._data:
            return self._data[QuADKeys.id]
        else:
            return hash(self.question)


class QAS:
    def __init__(self, _data: list):
        if _data:
            self._data = _data
        else:
            self._data = [QA()._data]

    def __getitem__(self, index: int) -> QA:
        return QA(self._data[index])

    def __len__(self):
        return len(self._data)


class Paragraph:
    def __init__(self, _data: dict):
        if _data:
            self._data = _data
        else:
            self._data = {QuADKeys.qas: QAS()._data, QuADKeys.context: ""}

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
        if _data:
            self._data = _data
        else:
            self._data = [Paragraph()._data]

    def __getitem__(self, index: int) -> Paragraph:
        return Paragraph(self._data[index])

    def __len__(self):
        return len(self._data)


class QuadData:
    def __init__(self, _data: list):
        if _data:
            self._data = _data
        else:
            self._data = [{QuADKeys.paragraphs: Paragraphs()._data}]

    def __getitem__(self, index: int) -> Paragraphs:
        return Paragraphs(self._data[index][QuADKeys.paragraphs])

    def __len__(self):
        return len(self._data)


class QUAD:
    """
    An Object view on the QuAD structure
    """

    pass
