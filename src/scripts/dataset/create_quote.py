from tqdm import tqdm

from src.io.filepaths import Datasets
from src.qa.dataset import Dataset, Articles, Paragraph, QA, Answer
from src.tar.translate import Translator
from src.utils.logging import get_logger

logger = get_logger(__file__)


def add_quotes(context, answer):
    quote_start_index = answer.answer_start
    quote_end_index = answer.answer_start + len(answer.text)
    return f'{context[:quote_start_index]}"{answer.text}"{context[quote_end_index:]}'


def main():
    squad: Dataset = Dataset.Squad1.DEV  # todo change to TRAIN after verifing it works
    quote: Dataset = Dataset(data=[])
    t = Translator()
    successes = 0
    fails = 0

    for en_articles in tqdm(squad.data):
        de_articles = Articles(paragraphs=[])
        for en_paragraph in en_articles.paragraphs:
            context = en_paragraph.context
            for en_qa in en_paragraph.qas:
                # get question, answer, context tripple
                en_question = en_qa.question
                en_answer = en_qa.answers[0]
                en_q_context = add_quotes(context, en_answer)

                # translate tripple
                de_question = t.en2de(en_question)
                de_answer = t.en2de(en_answer.text)
                de_answer_q = '"' + de_answer + '"'
                de_context = t.en2de(en_q_context)

                # retrieve quoted answer and remove quotes
                if de_context.count(de_answer_q) == 1:
                    answer_start = de_context.find(de_answer_q)
                    de_context.replace(de_answer_q, de_answer)
                # retrieve answer without quotes
                elif de_context.count(de_answer) == 1:
                    answer_start = de_context.find(de_answer)
                # skip this QA Pair because answer could not be retrieved
                # either not found or not unique
                else:
                    fails += 1
                    logger.info(f"Could not find (unique) answer {fails}")
                    continue

                successes += 1
                # add datapoint
                de_articles.paragraphs.append(
                    Paragraph(
                        context=de_context,
                        qas=[
                            QA(
                                question=de_question,
                                answers=[
                                    Answer(text=de_answer, answer_start=answer_start)
                                ],
                                id=en_qa.id,
                            )
                        ],
                    )
                )
        quote.data.append(de_articles)

    print(f"Done - ({successes}/{fails})")
    quote.save(Datasets.Squad1.Translated.Quote.TRAIN)


if __name__ == "__main__":
    main()
