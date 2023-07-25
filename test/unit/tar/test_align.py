from src.qa.quad import QUAD
from src.tar.retrive import retrive
from src.tar.tokenize import Tokenizer
from src.tar.utils import Span
from src.utils.logging import get_logger

logger = get_logger(__name__)


def test_surface_token_mapping():
    text = (
        "Madam President, I would like to confine my remarks to Alzheimer's "
        "disease .")
    tokens = [
        'Mada', 'm', 'President', ',', 'I', 'would', 'like', 'to', 'confi',
        'ne', 'my', 're', 'marks', 'to', 'Alzheimer', "'", 's', 'disease', '',
        '.']
    indexed_tokens = [(i, token) for i, token in enumerate(tokens)]

    gold_mapping = {
        (0, "Mada"): Span(0, 4),
        (1, "m"): Span(4, 5),
        (2, "President"): Span(6, 15),
        (3, ","): Span(15, 16),
        (4, "I"): Span(17, 18),
        (5, "would"): Span(19, 24),
        (6, "like"): Span(25, 29),
        (7, "to"): Span(30, 32),
        (8, "confi"): Span(33, 38),
        (9, "ne"): Span(38, 40),
        (10, "my"): Span(41, 43),
        (11, "re"): Span(44, 46),
        (12, "marks"): Span(46, 51),
        (13, "to"): Span(52, 54),
        (14, "Alzheimer"): Span(55, 64),
        (15, "'"): Span(64, 65),
        (16, "s"): Span(65, 66),
        (17, "disease"): Span(67, 74),
        (18, ""): Span(75, 75),
        (19, "."): Span(75, 76)
    }

    mapping = Tokenizer.surface_token_mapping(text, indexed_tokens)

    for (index, token), gold_span in gold_mapping:
        predicted_span = mapping[index]
        logger.info(f"({index}){token}, Span={predicted_span}")
        assert gold_span == predicted_span


def test_answer_extraction():
    def extract_suitable_test_pairs(source_dataset, target_dataset):
        for source_article, target_article in zip(
                source_dataset.data, target_dataset.data):
            for source_paragraph, target_paragraph in zip(
                    source_article, target_article):
                # extract context
                source_text = source_paragraph.context
                target_text = target_paragraph.context
                # estract answers
                for source_qa, target_qa in zip(
                        source_paragraph.qas, target_paragraph.qas):
                    for source_answer, target_answer in zip(
                            source_qa.answers, target_qa.answers):
                        # only take ansers where the answer text appears once
                        if target_text.count(target_answer.text) == 1:
                            yield (source_text, source_answer,
                                   target_text, target_answer)

    squad = QUAD(QUAD.Datasets.Squad1.TRAIN)
    raw_squad = QUAD(QUAD.Datasets.Squad1.Translated.Raw.TRAIN)

    for test_pairs in extract_suitable_test_pairs(squad, raw_squad):
        source_text, source_answer, target_text, target_answer = test_pairs
        logger.info(f"\n ====== Test Case: ====== \n"
                    f"\n ====== Source text: ====== \n"
                    f"{source_text}\n"
                    f"\n ====== Source answer: ====== \n"
                    f"{source_answer.text}\n"
                    f"\n ====== Target text: ====== \n"
                    f"{target_text}\n"
                    f"\n ====== Target answer: ====== \n"
                    f"{target_answer.text}\n")
        retrived_span = retrive(
            source_text, Span.from_answer(source_answer), target_text)
        logger.info(f"{retrived_span=}")
        logger.info(f"\n====== Extracted answer ======\n"
                    f"{retrived_span(target_text)}")

        # fix target_span, at this time (2023-07-21) the answer.answer_start
        # values are not correctly set
        target_span = Span(
            start=target_text.find(target_answer.text),
            end=target_text.find(target_answer.text) + len(target_answer.text))

        assert retrived_span == target_span
