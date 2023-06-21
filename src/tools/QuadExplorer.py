import json
from src.tools.io import to_json

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


class Answer:
    text: str
    answer_start: int = 0

    def __init__(self, answer: dict):
        self.text = answer[QuADKeys.text]

    def to_dict(self) -> dict:
        return {
                QuADKeys.text: self.text
                }


class QA:
    question: str
    answers: list[Answer]

    def __init__(self, qa: dict):
        self.question = qa[QuADKeys.question]
        self.answers = [Answer(answer) for answer in qa[QuADKeys.answers]]

    def to_dict(self) -> dict:
        return {
                QuADKeys.question: self.question,
                QuADKeys.answers: [answer.to_dict() for answer in self.answers]
                }


class Paragraph:
    context: str
    qas: list[QA]

    def __init__(self, paragraph: dict):
        self.context = paragraph[QuADKeys.context]
        self.qas = [QA(qa) for qa in paragraph[QuADKeys.qas]]

    def to_dict(self) -> dict:
        return {
                QuADKeys.context: self.context,
                QuADKeys.qas: [qa.to_dict() for qa in self.qas]
                }


class QuAD:
    paragraphs: list[Paragraph]

    class Datasets:
        GermanQuADTest = "./data/datasets/GermanQuAD/GermanQuAD_test.json"
        GermanQuADTrain = "./data/datasets/GermanQuAD/GermanQuAD_train.json"
        MLQA = "./data/datasets/MLQA/test-context-de-question-de.json"
        XQuAD = "./data/datasets/XQuAD/xquad.de.json"

    def __init__(self, path: str = "", paragraphs: list[Paragraph] = None):
        if path:
            data_dict = self.load(path)
            self.paragraphs = [Paragraph(p[QuADKeys.paragraphs][0]) for p in data_dict[QuADKeys.data]]
        if paragraphs:
            self.paragraphs = paragraphs

    def to_dict(self) -> dict:
        return {QuADKeys.paragraphs: [[paragraph.to_dict() for paragraph in self.paragraphs]]}

    @staticmethod
    def load(path: str) -> dict:
        with open(path, mode="r", encoding="utf-8") as f_in:
            return json.load(f_in)

    def save(self, path: str, version: str = ""):
        print(f"saving dataset '{version}' of size: '{len(self.paragraphs)}' to path: '{path}'")
        data = {}
        if version:
            data[QuADKeys.version] = version
        data = {QuADKeys.data: self.to_dict()}
        to_json(data, path)
