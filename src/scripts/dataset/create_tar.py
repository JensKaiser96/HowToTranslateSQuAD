import re

from src.io.filepaths import Datasets
from src.nlp_tools.span import Span
from src.qa.dataset import Dataset, Article, Paragraph, QA, Answer
from src.tar.retrive import retrieve
from src.utils.logging import get_logger

logger = get_logger(__file__)


def tar(context, answer):
    """
    returns the answer start index using the TAR method
    """
    possible_spans = [Span(*match.span()) for match in re.finditer(answer, context)]
    if len(possible_spans) == 0:
        return retrieve(source_text=, source_span=, target_text=context).start
    if len(possible_spans) == 1:
        return possible_spans.pop().start
    # len > 1
    return next_best(possible_spans, context, answer)


def main():
    raw: Dataset = Dataset.Raw.TRAIN
    squad: Dataset = Dataset.Squad1.TRAIN
    dataset = Dataset(data=[])
    for article_no, article in enumerate(raw.data):
        tar_article = Article(paragraphs=[])
        for paragraph_no, paragraph in enumerate(article.paragraphs):
            tar_paragraph = Paragraph(context="", qas=[])
            context = paragraph.context
            for qa_no, qa in enumerate(paragraph.qas):
                tar_qa = QA(question="", answers=[], id=qa.id)
                answer = qa.answers[0]
                possible_spans = [Span(*match.span()) for match in re.finditer(answer, context)]
                if len(possible_spans) == 1:
                    answer["answer_start"] = possible_spans.pop().start
                else:
                    squad_paragraph = squad.data[article_no].paragraphs[paragraph_no]
                    source_context = squad_paragraph.context
                    source_answer_start = squad_paragraph.qas[qa_no].answers[0].answer_start
                    answer["answer_start"] = tar(context, answer["text"])
                tar_paragraph.qas.append(tar_qa)
            if tar_paragraph.qas:
                tar_article.paragraphs.append(tar_paragraph)
        dataset.data.append(tar_article)
    dataset.save(
        Datasets.Squad1.Translated.Tar.TRAIN,
        version="TAR: v1. do alignment sentence wise, e.g. use nltk to split both target, and source into list of "
        "sentences, if the lists have different length, omit the extra sentences of the longer one. Seems to "
        "work 50% of the time (total: 672, success: 378, soft: 77, hard: 217) soft means one span is a sub span"
        " of the other",
    )


if __name__ == "__main__":
    main()
