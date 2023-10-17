import sys

import tqdm

from src.io.filepaths import Datasets
from src.qa.dataset import Dataset
from src.qa.qamodel import QAModel
from src.qa.squad_eval_script import compute_f1
from src.tar.translate import Translator
from src.utils.logging import get_logger

logger = get_logger(__file__, script=True)


def main(trg_ds: Dataset):
    threshold = 2/3
    english_qa_model: QAModel = QAModel.EnglishQA

    src_ds: Dataset = Dataset.Squad1.TRAIN
    back_ds: Dataset = Dataset(data=[])

    translator = Translator()

    for trg_article in tqdm.tqdm(trg_ds.data, position=1):
        for trg_paragraph in tqdm.tqdm(trg_article.paragraphs, position=0):
            back_context = translator.de2en(trg_paragraph.context)
            for trg_qa in trg_paragraph.qas:
                back_question = translator.en2de(trg_qa.question)
                back_answer = translator.en2de(trg_qa.answers[0].text)

                src_article_no, src_paragraph_no, src_qa_no = src_ds.get_qa_by_id(trg_qa.id)
                src_cqa = src_ds.data[src_article_no].paragraphs[src_paragraph_no]
                src_context = src_cqa.context
                src_question = src_cqa.qas[src_qa_no].question
                src_answer = src_cqa.qas[src_qa_no].answers[0].text

                # like thats ever going to happen, but hey
                if src_context == back_context and src_question == back_question and src_answer == back_answer:
                    back_ds.add_cqa_tuple(trg_paragraph.context, trg_qa.question, trg_qa.answers[0], trg_qa.id)
                orignial_prediction = english_qa_model.prompt(src_question, src_context)
                back_prediction = english_qa_model.prompt(back_question, back_context)
                if compute_f1(orignial_prediction.text, back_prediction.text) >= threshold:
                    back_ds.add_cqa_tuple(trg_paragraph.context, trg_qa.question, trg_qa.answers[0], trg_qa.id)

    back_ds.save(Datasets.Squad1.Translated.Raw.TRAIN_BACK, "raw-back")  # todo, set according to dataset


if __name__ == "__main__":
    if len(sys.argv) == 2:
        fuzzy_dataset_name = sys.argv[1]
        dataset = Dataset.from_fuzzy(fuzzy_dataset_name)
        main(dataset)
    else:
        logger.info("Please specify the dataset you want to run trough back as the first and only arg")
