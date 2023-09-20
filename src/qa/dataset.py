from typing import Optional

from pydantic import BaseModel

from src.io.filepaths import Datasets, StressTest, DATASETS_PATH
from src.io.utils import to_json
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Answer(BaseModel):
    text: str
    answer_start: int


class QA(BaseModel):
    question: str
    answers: list[Answer]
    id: Optional[str]


class Paragraph(BaseModel):
    context: str
    qas: list[QA]


class Article(BaseModel):
    title: Optional[str]
    paragraphs: list[Paragraph]


class Dataset(BaseModel):
    version: Optional[str]
    data: list[Article]
    path: Optional[str]

    class Squad1:
        @classmethod
        @property
        def TRAIN(cls):
            return Dataset.load(Datasets.Squad1.TRAIN)

        @classmethod
        @property
        def DEV(cls):
            return Dataset.load(Datasets.Squad1.DEV)

        @classmethod
        @property
        def TRAIN_SMALL(cls):
            return Dataset.load(Datasets.Squad1.TRAIN_SMALL)

    class GermanQUAD:
        @classmethod
        @property
        def SMALL(cls):
            return Dataset.load(Datasets.GermanQuad.SMALL)

        @classmethod
        @property
        def TRAIN(cls):
            return Dataset.load(Datasets.GermanQuad.TRAIN)

        @classmethod
        @property
        def TEST(cls):
            return Dataset.load(Datasets.GermanQuad.TEST)

    class MLQA:
        @classmethod
        @property
        def TEST(cls):
            return Dataset.load(Datasets.Mlqa.TEST)

    class XQUAD:
        @classmethod
        @property
        def TEST(self):
            return Dataset.load(Datasets.Xquad.TEST)

    class Raw:
        @classmethod
        @property
        def TRAIN(cls):
            return Dataset.load(Datasets.Squad1.Translated.Raw.TRAIN)

        @classmethod
        @property
        def TRAIN_CLEAN(cls):
            return Dataset.load(Datasets.Squad1.Translated.Raw.TRAIN_CLEAN)

    class StressTest:
        class Base:
            @classmethod
            @property
            def DIS(cls):
                return Dataset.load(StressTest.Base.DIS)

            @classmethod
            @property
            def NOT(cls):
                return Dataset.load(StressTest.Base.NOT)

            @classmethod
            @property
            def ONE(cls):
                return Dataset.load(StressTest.Base.ONE)

        @classmethod
        @property
        def DIS(cls):
            return Dataset.load(StressTest.DIS)

        @classmethod
        @property
        def NOT(cls):
            return Dataset.load(StressTest.NOT)

        @classmethod
        @property
        def ONE(cls):
            return Dataset.load(StressTest.ONE)

        @classmethod
        @property
        def OOD(cls):
            return Dataset.load(StressTest.OOD)

    @classmethod
    def load(cls, path: str) -> "Dataset":
        dataset: "Dataset" = cls.parse_file(path)
        dataset.path = path
        return dataset

    def save(self, path: str, version: str = ""):
        logger.info(
            f"saving dataset '{version}' of size: '{len(self.data)} to path: '{path}'"
        )
        if version:
            self.version = version
        to_json(self.json(indent=4), path)

    @property
    def name(self):
        return (
            self.path.removeprefix(DATASETS_PATH)
            .removesuffix(".json")
            .replace("/", ".")
        )

    def as_hf_dataset(self, tokenizer, max_length, split: str = "train"):
        """
        returns the dataset defined at the path as a HuggingFace Dataset. Note this completely ignores the content of
        the Dataset.load Object, only the data saved to the path is loaded.
        """
        import datasets
        from src.qa.train_util import prepare_train_features, flatten_quad

        if not self.path:
            raise AttributeError(
                "No path to load from specified. The HuggingFace dataset is loaded directly from the "
                "file, and not the actual Dataset.load Object"
            )
        raw_dataset = datasets.load_dataset(
            "json", data_files=self.path, field="data", split=split
        )
        flatt_dataset = raw_dataset.map(
            flatten_quad, batched=True, remove_columns=raw_dataset.column_names
        )
        tokenized_dataset = flatt_dataset.map(
            prepare_train_features,
            batched=True,
            remove_columns=flatt_dataset.column_names,
            fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
        )
        tokenized_dataset.set_format("torch")
        return tokenized_dataset


if __name__ == "__main__":
    Dataset.load("./../../" + Datasets.Squad1.TRAIN)
