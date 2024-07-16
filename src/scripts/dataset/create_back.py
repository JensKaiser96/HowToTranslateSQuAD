import sys

import tqdm

from src.io.filepaths import Datasets, Models
from src.qa.dataset import Dataset
from src.qa.qamodel import QAModel
from src.qa.squad_eval_script import compute_f1
from src.tar.translate import Translator
from src.utils.logging import get_logger

logger = get_logger(__file__, script=True)


def main(trg_ds: Dataset):
    threshold = 2/3  # this threshold is way too high
    english_qa_model = QAModel(Models.QA.Distilbert.ENGLISH_QA)

    src_ds = Dataset.load(Datasets.SQuAD.TRAIN)
    back_ds = Dataset()

    translator = Translator()

    stats = {"pass": 0, "fail": 0}

    for trg_article in tqdm.tqdm(trg_ds.data, position=1):
        for trg_paragraph in (pbar := tqdm.tqdm(trg_article.paragraphs, position=0)):
            pbar.set_description(str(stats))
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
                    stats["pass"] += 1
                    back_ds.add_cqa_tuple(trg_paragraph.context, trg_qa.question, trg_qa.answers[0], trg_qa.id)
                else:
                    orignial_prediction = english_qa_model.prompt(src_question, src_context)
                    back_prediction = english_qa_model.prompt(back_question, back_context)
                    if compute_f1(orignial_prediction.text, back_prediction.text) >= threshold:
                        stats["pass"] += 1
                        back_ds.add_cqa_tuple(trg_paragraph.context, trg_qa.question, trg_qa.answers[0], trg_qa.id)
                    else:
                        stats["fail"] += 1

    back_ds.save(Datasets.SQuAD.Translated.Quote.TRAIN_BACK, f"quote-back {stats=}")  # todo, set according to dataset


if __name__ == "__main__":
    if len(sys.argv) == 2:
        fuzzy_dataset_name = sys.argv[1]
        dataset = Dataset.from_fuzzy(fuzzy_dataset_name)
        main(dataset)
    else:
        logger.info("Please specify the dataset you want to run trough back as the first and only arg")
