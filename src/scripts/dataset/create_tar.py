import re

from src.io.filepaths import Datasets
from src.nlp_tools.span import Span
from src.qa.dataset import Dataset, Article, Paragraph, QA, Answer
from src.tar.retrive import retrieve
from src.utils.logging import get_logger

logger = get_logger(__file__)


def tar(src_context: str, src_answer: Answer, trg_context: str, trg_answer: Answer):
    """
    returns the answer start index using the TAR method
    """
    possible_spans = [Span(*match.span()) for match in re.finditer(trg_answer.text, trg_context)]

    if len(possible_spans) == 1:
        answer_span = possible_spans.pop()
    else:
        retrieved_answer_span = retrieve(
            source_text=src_context,
            source_span=Span.from_answer(src_answer),
            target_text=trg_context
        )
        if len(possible_spans) == 0:
            answer_span = retrieved_answer_span
        else:
            # sort after smallest difference to retrieved_answer_span and take first element
            answer_span = sorted(possible_spans, key=lambda element: retrieved_answer_span.compare(element)).pop()
    return Answer(answer_start=answer_span.start, text=answer_span(trg_context))


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
                answer = qa.answers[0]
                possible_spans = [Span(*match.span()) for match in re.finditer(answer, context)]
                # trivial
                if len(possible_spans) == 1:
                    retrieved_answer = Answer(text=answer, answer_start=possible_spans.pop().start)
                else:  # use tar
                    # load source references
                    squad_paragraph = squad.data[article_no].paragraphs[paragraph_no]
                    source_context = squad_paragraph.context
                    source_answer = squad_paragraph.qas[qa_no].answers[0]
                    # retrieve answer using TAR, AR
                    retrieved_answer = tar(source_context, source_answer, context, answer)
                tar_paragraph.qas.append(
                    QA(question=qa.question, answers=[retrieved_answer], id=qa.id)
                )
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
