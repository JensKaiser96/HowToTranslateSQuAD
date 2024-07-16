import os.path
from typing import Optional
from pathlib import Path

from pydantic import BaseModel, PrivateAttr, Field

from src.io.filepaths import Datasets, DATASETS, RESULTS
from src.io.utils import to_json
from src.nlp_tools.fuzzy import fuzzy_match
from src.qa.evaluate_dataset import DatasetEvaluation, get_dataset_evaluation
from src.utils.misc import get_inner_fields_recursive
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
    version: str = ""
    data: list[Article] = Field(default_factory=list)
    # fields which will not be written to file
    _path: Path = PrivateAttr(default=Path(DATASETS / "unnamed.json"))
    _qa_by_id: dict = PrivateAttr(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "Dataset":
        dataset: "Dataset" = cls.parse_file(path)
        dataset._path = path
        return dataset

    @property
    def name(self):
        return self._path.relative_to(DATASETS).with_suffix('').as_posix().replace('/', '.')

    @classmethod
    def get_dataset_names(cls) -> dict:
        return get_inner_fields_recursive(Datasets)

    @classmethod
    def from_fuzzy(cls, fuzzy_name: str):
        datasets = cls.get_dataset_names()
        dataset_name = fuzzy_match(fuzzy_name, list(datasets.keys()))
        if dataset_name is None:
            raise ValueError(f"Could not find definite match for '{fuzzy_name}'")
        logger.info(f"Loading Dataset {dataset_name}")
        return cls.load(datasets[dataset_name])

    def save(self, path: str, version: str = ""):
        logger.info(f"saving dataset '{version}' of size: '{len(self.data)} to path: '{path}'")
        if version:
            self.version = version
        to_json(self.json(indent=4, ensure_ascii=False), path)

    def get_qa_by_id(self, _id) -> tuple[int, int, int]:
        if not self._qa_by_id:
            self._generate_qa_id_dict()
        return self._qa_by_id[_id]

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
        return (RESULTS / "datasets" / self.name).with_suffix("json")

    def has_evaluation_file(self):
        return os.path.isfile(self.evaluation_path())

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

        if not self._path:
            raise AttributeError(
                "No path to load from specified. The HuggingFace dataset is loaded directly from the "
                "file, and not the actual Dataset.load Object"
            )
        raw_dataset = datasets.load_dataset(
            "json", data_files=self._path, field="data", split=split
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
