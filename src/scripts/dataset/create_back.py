import tqdm

from src.io.filepaths import Datasets
from src.qa.dataset import Dataset
from src.tar.translate import Translator
from src.utils.logging import get_logger

logger = get_logger(__file__, script=True)


# load raw_full
"""
WTF is back,

proposal: (c, q, a) =src2trg=> (...) =trg2src=> (c', q', a')
          m(c, q) => a'' , a'' == a ???
          
Goal:
    Check if translating was successful, 

"""


def main():
    src_ds: Dataset = Dataset.Squad1.TRAIN
    trg_ds: Dataset = Dataset.Raw.TRAIN

    translator = Translator()

    for src_article, trg_article in tqdm.tqdm(zip(src_ds.data, trg_ds.data), position=0):
        for src_paragraph, trg_paragraph in tqdm.tqdm(zip(src_article, trg_article), position=1):
            back_context = translator.de2en(trg_paragraph.context)
            for src_qa, trg_qa in zip(src_paragraph.qas, trg_paragraph.qas):
                if src_qa.id != trg_qa.id:
                    raise ValueError(f"IDs are not matching. {src_qa.id = }, {trg_qa.id}")
                back_question = translator.en2de(trg_qa.question)
                for src_answer, trg_answer in zip(src_qa.answers, trg_qa.answers):
                    back_answer = translator.en2de(trg_answer)

                    if src_paragraph.context == back_context and src_qa.question == back_question and src_answer.text == back_answer:
                        pass
                    answer.text = translator.en2de(answer.text)
    translated.save(Datasets.Squad1.Translated.Raw.TRAIN, "raw")


if __name__ == "__main__":
    main()
