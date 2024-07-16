from collections import Counter

from src.io.filepaths import Paths
from src.nlp_tools.words import get_question_type, get_answer_type, get_answers_type
from src.plot import plot_3bars, plot_51bars
from src.qa.dataset import Dataset
from src.qa.qamodel import QAModel


def plot_question_types(dataset_question_type_counter, raw_correct_question_type_counter, tar_correct_question_type_counter, quote_correct_question_type_counter):
    keys = list(dataset_question_type_counter.keys())

    print(dataset_question_type_counter)
    raw_values = [raw_correct_question_type_counter[key] / dataset_question_type_counter[key] for key in keys]
    tar_values = [tar_correct_question_type_counter[key] / dataset_question_type_counter[key] for key in keys]
    quote_values = [quote_correct_question_type_counter[key] / dataset_question_type_counter[key] for key in keys]

    save_path = Paths.PLOTS + "model_analysis/question_types"
    plot_3bars(keys, raw_values, tar_values, quote_values, save_path)


def plot_answer_types(model_name, answer_type_counter, correct_answer_type_counter):
    keys = ["date", "number", "capital", "lower", None]
    # values date, has all the times the prediction was date

    values = {key1: [answer_type_counter[(key2, key1)] / sum([answer_type_counter[(key3, key1)] for key3 in keys]) for key2 in keys] for key1 in keys}
    big_values = [correct_answer_type_counter[key] / sum([answer_type_counter[(key, key1)] for key1 in keys]) for key in keys]

    save_path = Paths.PLOTS + f"model_analysis/answer_types_{model_name}"
    plot_51bars(keys, values, big_values, save_path)


def main():
    dataset = Dataset.GermanQUAD.DEV
    sota = QAModel.GermanQuad.get_evaluation(dataset)
    raw = QAModel.RawClean.get_evaluation(dataset)
    tar = QAModel.TAR.get_evaluation(dataset)
    quote = QAModel.QUOTE.get_evaluation(dataset)

    threshold = 2/3

    dataset_question_type_counter = Counter()
    raw_correct_question_type_counter = Counter()
    tar_correct_question_type_counter = Counter()
    quote_correct_question_type_counter = Counter()

    raw_answer_type_counter = Counter()
    raw_correct_answer_type_counter = Counter()
    tar_answer_type_counter = Counter()
    tar_correct_answer_type_counter = Counter()
    quote_answer_type_counter = Counter()
    quote_correct_answer_type_counter = Counter()

    for raw_answer, tar_answer, quote_answer in zip(raw.individual_results, tar.individual_results, quote.individual_results):
        if raw_answer.id != tar_answer.id != quote_answer:
            raise ValueError(f"QA ids not matching, raw.id: '{raw_answer.id}', tar.id: '{tar_answer.id}', "
                             f"quote.id: '{quote_answer.id}'")
        article_id, paragraph_id, qa_id = dataset.get_qa_by_id(raw_answer.id)
        qa = dataset.data[article_id].paragraphs[paragraph_id].qas[qa_id]
        question_type = get_question_type(qa.question)
        gold_answer_type = get_answers_type(qa.answers)

        # Question Types
        dataset_question_type_counter[question_type] += 1
        if raw_answer.F1 >= threshold:
            raw_correct_question_type_counter[question_type] += 1
        if tar_answer.F1 >= threshold:
            tar_correct_question_type_counter[question_type] += 1
        if quote_answer.F1 >= threshold:
            quote_correct_question_type_counter[question_type] += 1

        # Answer Types
        raw_answer_type = get_answer_type(raw_answer.model_output.text)
        raw_answer_type_counter[(gold_answer_type, raw_answer_type)] += 1
        if raw_answer.F1 >= threshold:
            raw_correct_answer_type_counter[gold_answer_type] += 1
        tar_answer_type = get_answer_type(tar_answer.model_output.text)
        tar_answer_type_counter[(gold_answer_type, tar_answer_type)] += 1
        if tar_answer.F1 >= threshold:
            tar_correct_answer_type_counter[gold_answer_type] += 1
        quote_answer_type = get_answer_type(quote_answer.model_output.text)
        quote_answer_type_counter[(gold_answer_type, quote_answer_type)] += 1
        if quote_answer.F1 >= threshold:
            quote_correct_answer_type_counter[gold_answer_type] += 1

    plot_question_types(dataset_question_type_counter, raw_correct_question_type_counter, tar_correct_question_type_counter, quote_correct_question_type_counter)
    plot_answer_types("raw", raw_answer_type_counter, raw_correct_answer_type_counter)
    plot_answer_types("tar", tar_answer_type_counter, tar_correct_answer_type_counter)
    plot_answer_types("quote", quote_answer_type_counter, quote_correct_answer_type_counter)


if __name__ == '__main__':
    main()
