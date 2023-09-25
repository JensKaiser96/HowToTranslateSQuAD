from pydantic import BaseModel
from scipy.special import softmax
from tqdm import tqdm

from src.io.utils import to_json
from src.qa.dataset import Answer
from src.qa.dataset import Dataset
from src.qa.squad_eval_script import (
    compute_exact,
    compute_f1,
    compute_f1_precision_recall,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ModelOutput(BaseModel):
    start_logits: list[float]
    end_logits: list[float]
    start_index: int
    end_index: int
    span: tuple[int, int]
    text: str


class Result(BaseModel):
    id: str
    model_output: ModelOutput
    answers: list[str]
    best_answer: str
    EM: float
    F1: float
    recall: float
    precision: float
    confidence_start: float
    confidence_end: float


class Evaluation(BaseModel):
    EM: float
    F1: float
    recall: float
    precision: float
    confidence_start: float
    confidence_end: float
    total: int

    individual_results: list[Result]

    def save(self, path: str):
        self.summarize_results()
        to_json(self.json(indent=4), path=path)

    @classmethod
    def load(cls, path: str) -> "Evaluation":
        return cls.parse_file(path)

    def summarize_results(self):
        self.total = len(self.individual_results)
        self.EM = self.calculate_average("EM")
        self.F1 = self.calculate_average("F1")
        self.recall = self.calculate_average("recall")
        self.precision = self.calculate_average("precision")
        self.confidence_start = self.calculate_average("confidence_start")
        self.confidence_end = self.calculate_average("confidence_end")

    def calculate_average(self, field):
        return round(
            sum([getattr(result, field) for result in self.individual_results])
            / self.total,
            5,
        )


def evaluate(model, dataset: Dataset):
    """
    generates predictions on the dataset, saves them to the out_file, and then calls the evaluation script on it
    partially stolen from: https://rajpurkar.github.io/SQuAD-explorer/ -> "Evaluation Script"
    """
    logger.info(f"Evaluating {model.name} on {dataset.name} ...")
    evaluation = Evaluation(
        EM=0,
        F1=0,
        recall=0,
        precision=0,
        confidence_start=0,
        confidence_end=0,
        total=0,
        individual_results=[],
    )
    for article in tqdm(dataset.data):
        for paragraph in article.paragraphs:
            context = paragraph.context
            for qa in paragraph.qas:
                prediction = model.prompt(qa.question, context)
                evaluation.individual_results.append(
                    processing(prediction, qa.answers, qa.id)
                )

    evaluation.summarize_results()
    evaluation.save(
        path=model.results_path(dataset.name),
    )
    logger.info(
        f"=== Results ===\n"
        f"EM:         {evaluation.EM}\n"
        f"F1:         {evaluation.F1}\n"
        f"recall:     {evaluation.recall}\n"
        f"precision:  {evaluation.precision}\n"
        f"confidence_start: {evaluation.confidence_start}\n"
        f"confidence_end: {evaluation.confidence_end}\n"
        f"total:      {evaluation.total}"
    )
    return evaluation


def processing(prediction: ModelOutput, gold_answers: list[Answer], _id):
    exact_scores = [compute_exact(a.text, prediction.text) for a in gold_answers]
    f1_scores = [compute_f1(a.text, prediction.text) for a in gold_answers]
    best_em = max(exact_scores)
    best_f1 = max(f1_scores)

    best_answer_index = f1_scores.index(best_f1)
    best_answer = gold_answers[best_answer_index].text

    _, precision, recall = compute_f1_precision_recall(best_answer, prediction.text)

    confidence_start = softmax(prediction.start_logits)[prediction.start_index]
    confidence_end = softmax(prediction.end_logits)[prediction.end_index]

    return Result(
        id=_id,
        model_output=prediction,
        answers=[answer.text for answer in gold_answers],
        best_answer=best_answer,
        EM=best_em,
        F1=best_f1,
        recall=recall,
        precision=precision,
        confidence_start=confidence_start,
        confidence_end=confidence_end,
    )
