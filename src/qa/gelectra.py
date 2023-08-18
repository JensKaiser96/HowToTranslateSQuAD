import os

import torch
from tqdm import tqdm
from transformers.models.electra.modeling_electra import ElectraForQuestionAnswering
from transformers.models.electra.tokenization_electra_fast import ElectraTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding

from src.io.filepaths import Models, PREDICTIONS_PATH
from src.io.utils import to_json
from src.nlp_tools.span import Span
from src.nlp_tools.token import Tokenizer, surface_token_mapping
from src.qa.quad import QUAD
from src.qa.squad_eval_script import normalize_answer, compute_exact, compute_f1
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Gelectra:
    def __init__(self, path: str):
        self.path = path
        logger.info(f"Loading model {self.name} ...")
        self.tokenizer = Tokenizer(ElectraTokenizerFast.from_pretrained(path))
        self.model = ElectraForQuestionAnswering.from_pretrained(path)

    @property
    def name(self):
        return Gelectra.path2name(self.path)

    @staticmethod
    def path2name(path:str):
        return ".".join(path.strip("/").split("/")[-2:])

    @classmethod
    @property
    def Base(cls):
        return Gelectra("deepset/gelectra-large")

    @classmethod
    @property
    def GermanQuad(cls):
        return Gelectra("deepset/gelectra-large-germanquad")

    @classmethod
    @property
    def RawClean(cls):
        return Gelectra(Models.QA.Gelectra.raw_clean)

    def evaluate(self, dataset: QUAD):
        """
        generates predictions on the dataset, saves them to the out_file, and then calls the evaluation script on it
        partially stolen from: https://rajpurkar.github.io/SQuAD-explorer/ -> "Evaluation Script"
        """
        logger.info(f"Evaluating {self.name} on {dataset.name} ...")
        predictions = {}
        exact_scores = {}
        f1_scores = {}
        for article in tqdm(dataset.data):
            for paragraph in article:
                context = paragraph.context
                for qa in paragraph.qas:
                    prediction = self.prompt(context, qa.question)
                    predictions[qa.id] = prediction
                    gold_answers = [
                        a.text for a in qa.answers if normalize_answer(a.text)
                    ]
                    if not gold_answers:
                        # For unanswerable questions, only correct answer is empty string
                        gold_answers = [""]
                    # Take max over all gold answers
                    exact_scores[qa.id] = max(
                        compute_exact(a, prediction["text"]) for a in gold_answers
                    )
                    f1_scores[qa.id] = max(
                        compute_f1(a, prediction["text"]) for a in gold_answers
                    )
        # compute total scores
        total = len(predictions)
        total_scores = {
            "EM": round(sum(exact_scores.values()) / total, 5),
            "F1": round(sum(f1_scores.values()) / total, 5),
            "total": total,
        }
        to_json(
            data={"scores": total_scores, "predictions": predictions},
            path=Gelectra.results_pathname(self.name, dataset.name),
        )
        logger.info(total_scores)
        return total_scores

    @staticmethod
    def results_pathname(model_name: str, dataset_name: str):
        return f"{PREDICTIONS_PATH}{model_name}_{dataset_name}.json"

    @staticmethod
    def has_results_file(model_path: str, dataset_name: str):
        return os.path.isfile(Gelectra.results_pathname(Gelectra.path2name(model_path), dataset_name))

    def prompt(self, context: str, question: str):
        model_input = self.tokenizer.encode(context, question)
        with torch.no_grad():
            output = self.model(**model_input)

        # get answer on token level
        answer_start_index = output.start_logits.argmax()
        answer_end_index = output.end_logits.argmax()

        # convert token to surface level
        context_token_ids, _ = self._split_encoding(model_input)
        context_tokens = self.tokenizer.decode(context_token_ids(model_input))
        mapping = surface_token_mapping(context, context_tokens, "#")
        span = Span.combine(mapping[answer_start_index : answer_end_index + 1])

        return {
            "start_logits": output.start_logits.flatten().tolist(),
            "end_logits": output.end_logits.flatten().tolist(),
            "start_index": int(answer_start_index),
            "end_index": int(answer_end_index),
            "surface_span": (span.start, span.end),
            "text": span(context),
        }

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
