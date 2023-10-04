import os

import torch
from transformers.models.electra.modeling_electra import ElectraForQuestionAnswering
from transformers.models.electra.tokenization_electra_fast import ElectraTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding

from src.io.filepaths import Models, PREDICTIONS_PATH
from src.nlp_tools.span import Span
from src.nlp_tools.token import Tokenizer
from src.qa.dataset import Dataset
from src.qa.evaluate import ModelOutput, Evaluation, evaluate
from src.utils.decorators import classproperty
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Gelectra:
    lazy_loading = False

    def __init__(self, path: str):
        self.path = path
        self.tokenizer = None
        self.model = None
        if not Gelectra.lazy_loading:
            self.load_weights()

    def load_weights(self):
        logger.info(f"Loading model {self.name} ...")
        self.tokenizer = Tokenizer(ElectraTokenizerFast.from_pretrained(self.path))
        self.model = ElectraForQuestionAnswering.from_pretrained(self.path)
        self.model.to("cuda:0")

    def get_evaluation(self, dataset: Dataset, redo=False) -> Evaluation:
        if not redo and self.has_results_file(dataset.name):
            logger.info("Found Evaluation file of model on provided Dataset."
                        "Loading existing Evaluation.")
            return Evaluation.load(self.results_path(dataset.name))
        return evaluate(self, dataset)

    @property
    def name(self):
        return Gelectra.path2name(self.path)

    @staticmethod
    def path2name(path: str):
        return ".".join(path.strip("/").split("/")[-2:])

    @classmethod
    @classproperty
    def Base(cls) -> "Gelectra":
        return Gelectra("deepset/gelectra-large")

    @classmethod
    @classproperty
    def GermanQuad(cls) -> "Gelectra":
        return Gelectra("deepset/gelectra-large-germanquad")

    @classmethod
    @classproperty
    def RawClean(cls) -> "Gelectra":
        return Gelectra(Models.QA.Gelectra.raw_clean)

    @classmethod
    @classproperty
    def RawClean1(cls) -> "Gelectra":
        return Gelectra(Models.QA.Gelectra.raw_clean_1)

    @classmethod
    @classproperty
    def RawClean2(cls) -> "Gelectra":
        return Gelectra(Models.QA.Gelectra.raw_clean_2)

    @classmethod
    @classproperty
    def RawClean3(cls) -> "Gelectra":
        return Gelectra(Models.QA.Gelectra.raw_clean_3)

    @classmethod
    @classproperty
    def RawClean4(cls) -> "Gelectra":
        return Gelectra(Models.QA.Gelectra.raw_clean_4)

    def results_path(self, dataset_name: str):
        return f"{PREDICTIONS_PATH}{self.name}_{dataset_name}.json"

    def has_results_file(self, dataset_name: str):
        return os.path.isfile(self.results_path(dataset_name))

    @staticmethod
    def filter_dict_for_model_input(input_dict: dict):
        valid_keys = ["input_ids", "token_type_ids", "attention_mask"]
        return {key: value for key, value in input_dict.items() if key in valid_keys}

    def prompt(self, question: str, context: str) -> ModelOutput:
        if self.model is None or self.tokenizer is None:
            self.load_weights()
        model_input = self.tokenizer.encode_qa(question, context)
        model_input.to("cuda:0")
        with torch.no_grad():
            output = self.model(**Gelectra.filter_dict_for_model_input(model_input))

        # get answer on token level
        # This works even on a tensor, the output of argmax() is its index as if it was 1D
        answer_start_token_index = int(output.start_logits.argmax())
        answer_end_token_index = int(output.end_logits.argmax())

        # get answer on surface level
        # flatten the whole vector except for the start and end pairs:
        # [[token_1_start, token_1_end], [token_2_start, token_2_end], ...]
        answer_start_surface_index = int(
            model_input.offset_mapping.reshape(-1, 2)[answer_start_token_index][0]
        )
        answer_end_surface_index = int(
            model_input.offset_mapping.reshape(-1, 2)[answer_end_token_index][1]
        )

        return ModelOutput(
            start_logits=output.start_logits.flatten().tolist(),
            end_logits=output.end_logits.flatten().tolist(),
            start_index=answer_start_token_index,
            end_index=answer_end_token_index,
            span=(answer_start_surface_index, answer_end_surface_index),
            text=context[answer_start_surface_index:answer_end_surface_index],
        )

    def _split_encoding(self, encoding: BatchEncoding) -> tuple[Span, Span]:
        """
        splits the combined encoding of the ElectraTokenizer of a context
        question pair into its two spans
            [[CLS, <context>, SEP, <question>, SEP]]
        """
        CLS = self.tokenizer.model.cls_token_id
        SEP = self.tokenizer.model.sep_token_id

        ids = list(encoding.input_ids.flatten())

        # check if sequence is as expected
        if ids[0] != CLS:
            raise ValueError(
                f"Expected sequence to start with [CLS] token (id:{CLS}), "
                f"but sequence starts with id:{ids[0]}."
            )
        if ids[-1] != SEP:
            raise ValueError(
                f"Expected sequence to end with [SEP] token (id:{SEP}), "
                f"but sequence ends with id:{ids[-1]}."
            )
        if list(ids).count(SEP) != 2:
            raise ValueError(
                f"Expected sequence to have exactly three occurences of the "
                f"[SEP] token (id:{SEP}), but counted {list(ids).count(SEP)} "
                f"instead."
            )

        first_pad = ids.index(SEP)

        context = Span(1, first_pad)
        question = Span(first_pad + 1, len(ids) - 1)

        # Verify spans are not empty
        if context.is_empty:
            raise ValueError(f"Source span is not allowed to be empty. {context}")
        if question.is_empty:
            raise ValueError(f"Target span is not allowed to be empty. {question}")

        return context, question
