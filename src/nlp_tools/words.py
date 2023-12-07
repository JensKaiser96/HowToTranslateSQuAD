import re

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
    "what": {"what", "what's"},
    "how": {"how"},
    "when": {"when"},
    "who": {"who", "whom", "whose", "who's"},
    "where": {"where"},
    "which": {"which"},
    "why": {"why"},
}

question_word_mapping_en_de = {
        "what": "was",
        "how": "wie",
        "when": "wann",
        "who": "wer",
        "where": "wo",
        "which": "welche",
        "why": "warum",
        "null": "null",
        None: None
    }

months_de = [
    "Januar",
    "Februar",
    "März",
    "April",
    "Mai",
    "Juni",
    "Juli",
    "August",
    "September",
    "Oktober",
    "November",
    "Dezember"
]
months_en = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"
]


def get_question_type(question: str, en=False, verbose=False):
    question_tokens = [token.lower() for token in get_tokens(question)]

    possible_types = []

    question_words = question_words_en if en else question_words_de
    for key, question_words in question_words.items():
        # Check if the first word is a question word
        if question_tokens[0] in question_words:
            return key
        # Search in the rest of the question for questions words
        for question_word in question_words:
            if question_word in question_tokens:
                possible_types.append(key)

    if len(possible_types) == 1:
        return possible_types.pop()

    else:
        if verbose:
            logger.info(f"Could not determine type of Question, found '{len(possible_types)} possible types': {question}")
        return None


def get_answer_type(answer: str, en=False):
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
            pattern = date_pattern.replace("months", " | ".join(months_en))
        else:
            pattern = date_pattern.replace("months", " | ".join(months_de))
        return bool(re.match(pattern, text))

    def all_capital_first(s):
        return all([word[0].isupper() for word in s.split()])

    def all_lower_first(s):
        return all([word[0].islower() for word in s.split()])

    if is_number(answer):
        return "number"
    elif contains_date(answer):
        return "date"
    elif all_capital_first(answer):
        return "capital"
    elif all_lower_first(answer):
        return "lower"
    return None


def get_answers_type(answers: list, en=False):
    types = []
    for answer in answers:
        type_ = get_answer_type(answer.text, en)
        if type_:
            types.append(type_)
    if len(types) == 1:
        return types.pop()
    return None


