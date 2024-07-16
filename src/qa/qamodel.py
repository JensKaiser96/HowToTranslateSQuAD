import os
from enum import Enum, auto
from pathlib import Path

from src.io.filepaths import Models, Paths
from src.utils.misc import get_inner_fields_recursive
from src.nlp_tools.fuzzy import fuzzy_match
from src.nlp_tools.span import Span
from src.nlp_tools.token import Tokenizer
from src.qa.dataset import Dataset
from src.qa.evaluate_predictions import ModelOutput, PredictionEvaluation, get_predictions_evaluation
from src.utils.logging import get_logger

logger = get_logger(__name__)


class QAModel:
    class ModelTypes(Enum):
        Gelectra = auto()
        DistilBert = auto()

        @classmethod
        def from_path(cls, path: Path) -> "ModelTypes":
            for model_type in cls:
                if model_type.name.lower() in path.as_posix().lower():
                    return model_type
            raise AttributeError(f"Could not infer model type from path: '{path}', path must contain one of {cls.__members__.keys()}")

    def __init__(self, path: Path, model_type: ModelTypes = None):
        self.path: Path = path
        self.tokenizer = None
        self.model = None
        if model_type is None:
            self.model_type = self.ModelTypes.from_path(path)
        else:
            self.type = model_type

    def load_weights(self) -> None:
        logger.info(f"Loading {self.type} model {self.name} ...")
        if self.type == self.ModelTypes.Gelectra:
            from transformers.models.electra.modeling_electra import ElectraForQuestionAnswering
            from transformers.models.electra.tokenization_electra_fast import ElectraTokenizerFast
            self.tokenizer = Tokenizer(ElectraTokenizerFast.from_pretrained(self.path))
            self.model = ElectraForQuestionAnswering.from_pretrained(self.path)
        elif self.type == self.ModelTypes.DistilBert:
            from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering
            self.tokenizer = Tokenizer(DistilBertTokenizerFast.from_pretrained(self.path))
            self.model = DistilBertForQuestionAnswering.from_pretrained(self.path)
        else:
            raise ValueError(f"Unknown QAModel type: '{self.type}'")

        self.model.to("cuda:0")

    def get_evaluation(self, dataset: Dataset, redo=False) -> PredictionEvaluation:
        if not redo and self.has_results_file(dataset.name):
            logger.info("Found Evaluation file of model on provided Dataset."
                        "Loading existing Evaluation.")
            return PredictionEvaluation.load(self.results_path(dataset.name))
        return get_predictions_evaluation(self, dataset)

    @classmethod
    def from_fuzzy(cls, fuzzy_name) -> "QAModel":
        models = cls.get_model_names()
        model_name = fuzzy_match(fuzzy_name, list(models.keys()))
        if model_name is None:
            raise ValueError(f"Could not find definite match for '{fuzzy_name}'")
        logger.info(f"Loading Model {model_name}")
        return QAModel(models[model_name])

    @property
    def name(self) -> str:
        return self.path.relative_to(Paths.MODELS).with_suffix('').as_posix().replace('/', '.')

    @classmethod
    def get_model_names(cls) -> dict[str: Path]:
        return get_inner_fields_recursive(Models.QA)

    @classmethod
    def get_all_models(cls) -> list["QAModel"]:
        models = [cls(model_path) for model_name, model_path in cls.get_model_names().items() if model_name != "Base"]
        return models

    def results_path(self, dataset_name: str) -> Path:
        return Paths.RESULTS / "models" / f"{self.name}_{dataset_name}.json"

    def has_results_file(self, dataset_name: str) -> bool:
        return os.path.isfile(self.results_path(dataset_name))

    @staticmethod
    def filter_dict_for_model_input(input_dict: dict):
        valid_keys = ["input_ids", "token_type_ids", "attention_mask"]
        return {key: value for key, value in input_dict.items() if key in valid_keys}

    def prompt(self, question: str, context: str) -> ModelOutput:
        import torch
        if self.model is None or self.tokenizer is None:
            self.load_weights()
        model_input = self.tokenizer.encode_qa(question, context)
        with torch.no_grad():
            output = self.model(**QAModel.filter_dict_for_model_input(model_input))

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

    def _split_encoding(self, encoding) -> tuple[Span, Span]:
        """
        splits the combined encoding of the ElectraTokenizer of a context
        question pair into its two spans
            [[CLS, <context>, SEP, <question>, SEP]]
        """
        from transformers.tokenization_utils_base import BatchEncoding
        encoding: BatchEncoding

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
