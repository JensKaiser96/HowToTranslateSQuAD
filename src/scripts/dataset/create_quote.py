
import re

from tqdm import tqdm

from src.io.filepaths import Datasets
from src.nlp_tools.fuzzy import fuzzy_match
from src.qa.dataset import Dataset, Article, Paragraph, QA, Answer
from src.tar.translate import Translator
from src.utils.logging import get_logger

logger = get_logger(__file__)

# ''  9 / 11
# "" 12 / 8
# () 12 / 8
# article mismatch, no translation
quote_start_symbol = '"'
quote_end_symbol = quote_start_symbol


def quote(text: str):
    return quote_start_symbol + text + quote_end_symbol


def add_quotes(context, answer):
    quote_start_index = answer.answer_start
    quote_end_index = answer.answer_start + len(answer.text)
    return context[:quote_start_index] + quote(answer.text) + context[quote_end_index:]


def new_paragraph(context, question, answer, answer_start, _id) -> Paragraph:
    return Paragraph(
        context=context,
        qas=[
            QA(
                question=question,
                answers=[Answer(text=answer, answer_start=answer_start)],
                id=_id,
            )
        ],
    )


def extreact_between_quotes(text: str) -> list[str]:
    pattern = r'"([^"]*)"'
    return [m.group(1) for m in re.finditer(pattern, text)]


def retrieve_answer(de_context, de_answer, en_context, en_answer, debug=False):
    de_answer_q = quote(de_answer)

    # happy path, answer is quoted in context
    if de_context.count(de_answer_q) == 1:
        answer_start = de_context.find(de_answer_q)
        de_context = de_context.replace(de_answer_q, de_answer)
        return answer_start, de_context, de_answer

    # still happy, answer is in context, (without quotes)
    if de_context.count(de_answer) == 1:
        answer_start = de_context.find(de_answer)
        return answer_start, de_context, de_answer

    # somewhat less happy, en_answer is in context
    if en_answer.count(de_answer) == 1:
        answer_start = de_context.find(en_answer)
        return answer_start, de_context, en_answer

    # last resort, take whatever matches best in between two quotes, but make sure the de_context has two more quotes than the english
    if en_context.count("\"") + 2 <= de_context.count("\""):
        options = extreact_between_quotes(de_context)
        best_answer = fuzzy_match(de_answer, options)
        if best_answer is not None:  # make sure best_answer is unique
            answer_start = de_context.find(best_answer)
            de_context = de_context.replace(quote(best_answer), best_answer)
            return answer_start, de_context, best_answer

    # skip this QA Pair because answer could not be retrieved
    # either not found or not unique
    if debug:
        print(f"Answer En: {en_answer}")
        print(f"Answer De: {de_answer}")
        print(f"Context DE:\n {de_context}")
    return None, None, None


def main():
    squad = Dataset.load(Datasets.SQuAD.TRAIN)
    quote_ds = Dataset()

    t = Translator()
    successes = 0
    fails = 0

    for en_articles in tqdm(squad.data, position=0):
        de_articles = Article(paragraphs=[])
        for en_paragraph in tqdm(en_articles.paragraphs, position=1):
            en_context = en_paragraph.context
            for en_qa in en_paragraph.qas:
                # get question, answer, context tripple
                en_question = en_qa.question
                en_answer = en_qa.answers[0]
                en_q_context = add_quotes(en_context, en_answer)

                # translate tripple
                de_question = t.en2de(en_question)
                de_answer = t.en2de(en_answer.text)
                de_context = t.en2de(en_q_context)

                answer_start, de_context, de_answer = retrieve_answer(de_context, de_answer, en_context, en_answer.text)
                if answer_start is None:
                    fails += 1
                else:
                    successes += 1
                    # add datapoint
                    de_articles.paragraphs.append(
                        new_paragraph(de_context, de_question, de_answer, answer_start, en_qa.id)
                    )
        quote_ds.data.append(de_articles)

        print(f"... - ({successes}/{fails+successes})")
    print(f"Done - ({successes}/{fails+successes})")
    quote_ds.save(Datasets.SQuAD.Translated.Quote.TRAIN)


if __name__ == "__main__":
    main()
