from src.nlp_tools.span import Span
from src.qa.quad import QUAD

from src.tar.retrive import retrieve
from src.utils.logging import get_logger

logger = get_logger(__name__)


def extract_suitable_test_pairs(source_dataset, target_dataset):
    for source_article, target_article in zip(source_dataset.data, target_dataset.data):
        for source_paragraph, target_paragraph in zip(source_article, target_article):
            # extract context
            source_text = source_paragraph.context
            target_text = target_paragraph.context
            # extract answers
            for source_qa, target_qa in zip(source_paragraph.qas, target_paragraph.qas):
                for source_answer, target_answer in zip(
                    source_qa.answers, target_qa.answers
                ):
                    # only take answers where the answer text appears once
                    if target_text.count(target_answer.text) == 1:
                        yield source_text, source_answer, target_text, target_answer


def test_answer_extraction():
    squad = QUAD.Squad1.TRAIN
    raw_squad = QUAD.Raw.TRAIN

    total_samples = 0
    successes = 0
    soft_fails = 0
    hard_fails = 0

    for test_pairs in extract_suitable_test_pairs(squad, raw_squad):
        total_samples += 1
        source_text, source_answer, target_text, target_answer = test_pairs
        # logger.info(
        #     f"\n ====== Test Case: ====== \n"
        #     f"\n ====== Source text: ====== \n"
        #     f"{source_text}\n"
        #     f"\n ====== Source answer: ====== \n"
        #     f"{source_answer.text}\n"
        #     f"\n ====== Target text: ====== \n"
        #     f"{target_text}\n"
        #     f"\n ====== Target answer: ====== \n"
        #     f"{target_answer.text}\n"
        # )
        retrived_span = retrieve(
            source_text, Span.from_answer(source_answer), target_text
        )
        # logger.info(f"{retrived_span=}")
        # logger.info(f"\n====== Extracted answer ======\n{retrived_span(target_text)}")

        # fix target_span, at this time (2023-07-21) the answer.answer_start
        # values are not correctly set
        target_span = Span(
            start=target_text.find(target_answer.text),
            end=target_text.find(target_answer.text) + len(target_answer.text),
        )

        if target_span == retrived_span:
            successes += 1
        elif target_span.is_subspan(retrived_span) or retrived_span.is_subspan(
            target_span
        ):
            soft_fails += 1
        else:
            hard_fails += 1

        # print(
        #     f"total: {total_samples}, success: {successes}, soft: {soft_fails}, hard: {hard_fails}"
        # )
