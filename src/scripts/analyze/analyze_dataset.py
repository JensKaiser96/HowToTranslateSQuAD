import re
import sys
from collections import Counter

from tqdm import tqdm

from src.nlp_tools.token import get_token_count
from src.qa.dataset import Dataset
from src.utils.logging import get_logger

logger = get_logger(__file__, script=True)


def record_question_type(question: str, question_words: list[str], counter: Counter):
    question_type = None
    for word in question_words:
        if word.lower() in question.lower():
            if question_type is not None:  # two different question words in question
                logger.info(f"Question has more than one question word: {question}")
                return None
            question_type = word
    counter[question_type] += 1


def record_answer_type(answer: str, counter: Counter):
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
        date_pattern = (
            r"\d{2}\.(?: Januar | Februar | März | April | Mai | Juni | Juli | August | September | "
            r"Oktober | November | Dezember |\d{2}\.)\d{4}"
        )
        return bool(re.match(date_pattern, text))

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


def main():
    question_words = [
        "Welche",
        "Was",
        "Wie",
        "Wann",
        "Wer",
        "Wie",
        "Wo",
        "Welche",
        "Warum",
    ]

    context_length_counter = Counter()
    question_type_counter = Counter()
    answer_type_counter = Counter()
    answer_length_counter = Counter()

    dataset = Dataset.from_fuzzy(fuzzy_name)
    for article in tqdm(dataset.data):
        for paragraph in article.paragraphs:
            context = paragraph.context
            context_length_counter[get_token_count(context)] += 1
            for qa in paragraph.qas:
                question = qa.question
                record_question_type(question, question_words, question_type_counter)
                for answer in qa.answers:
                    answer = answer.text
                    answer_length_counter[get_token_count(answer)] += 1
                    record_answer_type(answer, answer_type_counter)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        fuzzy_name = sys.argv[1]
        main()
    raise ValueError("Please specify a dataset to analyse.")
