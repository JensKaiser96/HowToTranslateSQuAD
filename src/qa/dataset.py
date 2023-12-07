import os.path
from typing import Optional

from pydantic import BaseModel, PrivateAttr

from src.io.filepaths import Datasets, StressTest, DATASETS_PATH, RESULTS_PATH
from src.io.utils import to_json
from src.nlp_tools.fuzzy import fuzzy_match
from src.qa.evaluate_dataset import DatasetEvaluation, get_dataset_evaluation
from src.utils.decorators import classproperty
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
    _qa_by_id: dict = PrivateAttr(default_factory=dict)

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
        @classproperty
        def TRAIN(cls) -> "Dataset":
            return Dataset.load(Datasets.GermanQuad.TRAIN)

        @classmethod
        @classproperty
        def TEST(cls) -> "Dataset":
            return Dataset.load(Datasets.GermanQuad.TEST)

        @classmethod
        @classproperty
        def DEV(cls) -> "Dataset":
            return Dataset.load(Datasets.GermanQuad.DEV)

        @classmethod
        @classproperty
        def TRAIN_WO_DEV(cls) -> "Dataset":
            return Dataset.load(Datasets.GermanQuad.TRAIN_WO_DEV)

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

    class Tar:
        @classmethod
        @property
        def TRAIN(cls):
            return Dataset.load(Datasets.Squad1.Translated.Tar.TRAIN)

    class Quote:
        @classmethod
        @property
        def TRAIN(cls):
            return Dataset.load(Datasets.Squad1.Translated.Quote.TRAIN)

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
    def get_dataset_names(cls):
        return [
            f"{attr}.{sub_attr}"
            for attr in dir(cls)
            if not attr.startswith('_') and attr[0].isupper()
            for sub_attr in dir(getattr(cls, attr))
            if not sub_attr.startswith('_') and sub_attr[0].isupper()
        ]

    @classmethod
    def load(cls, path: str) -> "Dataset":
        dataset: "Dataset" = cls.parse_file(path)
        dataset.path = path
        return dataset

    @classmethod
    def from_fuzzy(cls, fuzzy_name: str):
        dataset_name = fuzzy_match(fuzzy_name, cls.get_dataset_names())
        if dataset_name is None:
            raise ValueError(f"Could not find definite match for '{fuzzy_name}'")
        logger.info(f"Loading Dataset {dataset_name}")
        dataset_parent_name, dataset_child_name = dataset_name.split(".")
        dataset_parent = getattr(Dataset, dataset_parent_name)
        return getattr(dataset_parent, dataset_child_name)

    def save(self, path: str, version: str = ""):
        logger.info(
            f"saving dataset '{version}' of size: '{len(self.data)} to path: '{path}'"
        )
        if version:
            self.version = version
        to_json(self.json(indent=4, ensure_ascii=False), path)

    @property
    def name(self):
        return (
            self.path.removeprefix(DATASETS_PATH)
            .removesuffix(".json")
            .replace("/", ".")
        )

    def _generate_qa_id_dict(self):
        for article_no, article in enumerate(self.data):
            for paragraph_no, paragraph in enumerate(article.paragraphs):
                for qa_no, qa in enumerate(paragraph.qas):
                    self._qa_by_id[qa.id] = (article_no, paragraph_no, qa_no)

    def get_evaluation(self, redo=False) -> DatasetEvaluation:
        if not redo and self.has_evaluation_file():
            logger.info("Found Evaluation file of dataset, loading existing evaluation ...")
            return DatasetEvaluation.load(self.evaluation_path())
        return get_dataset_evaluation(self, self.name == self.Squad1.TRAIN.name)

    def evaluation_path(self):
        return f"{RESULTS_PATH}datasets/{self.name}.json"

    def has_evaluation_file(self):
        return os.path.isfile(self.evaluation_path())

    def get_qa_by_id(self, _id) -> tuple[int, int, int]:
        if not self._qa_by_id:
            self._generate_qa_id_dict()
        return self._qa_by_id[_id]

    def add_cqa_tuple(self, context: str, question: str, answer: Answer, _id: str):
        """
        In an optimal world there would be a check if the context is already in the dataset, and the qa pair would be
        added there instead of creating a whole new entry in the article/paragraph list.
        """
        self.data.append(
            Article(paragraphs=[
                Paragraph(context=context,
                          qas=[
                              QA(question=question,
                                 answers=[answer],
                                 id=_id)
                          ])
            ])
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
