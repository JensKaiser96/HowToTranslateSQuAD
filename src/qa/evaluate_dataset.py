from collections import defaultdict

from pydantic import BaseModel
from tqdm import tqdm

from src.io.utils import to_json
from src.nlp_tools.token import get_token_count
from src.nlp_tools.words import get_question_type, get_answer_type
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DatasetEvaluation(BaseModel):
    dataset_name: str
    number_qa_pairs: int
    question_types: defaultdict[str, int]
    answer_types: defaultdict[str, int]
    answer_lengths: list[int]
    context_lengths: list[int]

    def save(self, path: str):
        to_json(self.json(indent=4), path=path)

    @classmethod
    def load(cls, path: str) -> "DatasetEvaluation":
        return cls.parse_file(path)


def get_dataset_evaluation(dataset, en=False) -> DatasetEvaluation:
    dataset_evaluation = DatasetEvaluation(
        dataset_name=dataset.name,
        number_qa_pairs=0,
        question_types=defaultdict(int),
        answer_types=defaultdict(int),
        answer_lengths=[],
        context_lengths=[],
    )

    for article in tqdm(dataset.data):
        for paragraph in article.paragraphs:
            context = paragraph.context
            dataset_evaluation.context_lengths.append(get_token_count(context))
            for qa in paragraph.qas:
                valid_answer = False
                for answer in qa.answers:
                    if answer.text and answer.answer_start > 0:
                        valid_answer = True
                        dataset_evaluation.number_qa_pairs += 1
                        answer = answer.text
                        dataset_evaluation.answer_lengths.append(get_token_count(answer))
                        dataset_evaluation.answer_types[get_answer_type(answer, en=en)] += 1
                if valid_answer:
                    question = qa.question
                    dataset_evaluation.question_types[get_question_type(question, en=en)] += 1

    dataset_evaluation.save(dataset.evaluation_path())
    return dataset_evaluation
