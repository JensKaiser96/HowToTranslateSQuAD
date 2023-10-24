import re
from collections import defaultdict

from pydantic import BaseModel
from tqdm import tqdm

from src.io.utils import to_json
from src.nlp_tools.token import get_token_count
from src.qa.squad_eval_script import get_tokens
from src.utils.logging import get_logger

logger = get_logger(__name__)

question_words_de = {
    "was": {
        "was",
        "worauf",
        "wovon",
        "wodurch",
        "woraus",
        "woran",
        "wofür",
        "worüber",
        "worin",
        "worum",
        "womit",
        "wovor",
    },
    "wie": {
        "wie",
    },
    "wann": {
        "wann",
    },
    "wer": {"wer", "wen", "wem", "wessen"},
    "wo": {"wo", "woher", "wohin"},
    "welche": {"welche", "welch", "welcher", "welches", "welchem", "welchen"},
    "warum": {"warum", "wozu"},
}
question_words_en = {
    "what": {"what"},
    "how": {"how"},
    "when": {"when"},
    "who": {"who"},
    "where": {"where"},
    "which": {"which"},
    "why": {"why"},
}

months_de = "Januar Februar März April Mai Juni Juli August September Oktober November Dezember"
months_en = "January February March April May June July August September October November December"


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


def record_question_type(question: str, question_words_dict: dict[str, set[str]], counter: dict):
    question_tokens = [token.lower() for token in get_tokens(question)]

    possible_types = []

    for key, question_words in question_words_dict.items():
        # Check if the first word is a question word
        if question_tokens[0] in question_words:
            counter[key] += 1
            return
        # Search in the rest of the question for questions words
        for question_word in question_words:
            if question_word in question_tokens:
                possible_types.append(key)

    if len(possible_types) == 1:
        counter[possible_types.pop()] += 1
    else:
        counter[None] += 1
        logger.info(f"Could not determine type of Question, found '{len(possible_types)} possible types': {question}")


def record_answer_type(answer: str, counter: dict, en=False):
    """
    Date 8.9% 19 October 1512
    Other Numeric 10.9% 12
    Person 12.9% Thomas Coke
    Location 4.4% Germany
    Other Entity 15.3% ABC Sports
    Common Noun Phrase 31.8% property damage
    Adjective Phrase 3.9% second-largest
    Verb Phrase 5.5% returned to Earth
    Clause 3.7% to avoid trivialization
    Other 2.7% quietly
   """

    def is_number(text: str):
        return bool(re.match(r"^\d+$", text))

    def contains_date(text):
        date_pattern = r"\d{2}\.(?: months |\d{2}\.)\d{4}"
        if en:
            pattern = date_pattern.replace("months", months_en)
        else:
            pattern = date_pattern.replace("months", months_de)
        return bool(re.match(pattern, text))

    def all_capital_first(s):
        return all([word[0].isupper() for word in s.split()])

    def all_lower_first(s):
        return all([word[0].islower() for word in s.split()])

    if is_number(answer):
        counter["number"] += 1
    elif contains_date(answer):
        counter["date"] += 1
    elif all_capital_first(answer):
        counter["capital"] += 1
    elif all_lower_first(answer):
        counter["lower"] += 1


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
                        record_answer_type(answer, dataset_evaluation.answer_types, en=en)
                if valid_answer:
                    question = qa.question
                    if en:
                        qword_types = question_words_en
                    else:
                        qword_types = question_words_de
                    record_question_type(question, qword_types, dataset_evaluation.question_types)

    dataset_evaluation.save(dataset.evaluation_path())
    return dataset_evaluation
