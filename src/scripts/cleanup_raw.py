from src.qa.quad import QUAD
from src.utils.logging import get_logger

logger = get_logger(__file__)


def main():
    raw = QUAD(QUAD.Datasets.Squad1.Translated.Raw.TRAIN)
    raw_clean = QUAD()
    for article in raw.data._data:
        clean_article = article.copy()
        clean_article["paragraphs"] = []
        for paragraph in article["paragraphs"]:
            clean_paragraph = paragraph.copy()
            clean_paragraph["qas"] = []
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                clean_qa = qa.copy()
                answer = clean_qa["answers"][0]
                if context.count(answer["text"]) == 1:
                    # set correct answer_start
                    answer["answer_start"] = context.find(answer["text"])
                    clean_paragraph["qas"].append(clean_qa)
                else:
                    pass
            if clean_paragraph["qas"]:
                clean_article["paragraphs"].append(clean_paragraph)
        raw_clean.data._data.append(clean_article)
    raw_clean.save(QUAD.Datasets.Squad1.Translated.Raw.TRAIN_CLEAN, version="raw_clean")


if __name__ == "__main__":
    main()
