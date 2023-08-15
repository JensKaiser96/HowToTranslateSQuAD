import torch
from transformers.models.electra.modeling_electra import ElectraForQuestionAnswering
from transformers.models.electra.tokenization_electra_fast import ElectraTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding

from src.io.filepaths import Models
from src.nlp_tools.span import Span
from src.nlp_tools.token import Tokenizer, surface_token_mapping
from src.qa.quad import QUAD


class Gelectra:
    def __init__(self, name):
        self.tokenizer = Tokenizer(ElectraTokenizerFast.from_pretrained(name))
        self.model = ElectraForQuestionAnswering.from_pretrained(name)

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
        return Gelectra(Models.QA)

    def evaluate(self, Dataset: QUAD, out_file: str):
        """
        generates predictions on the dataset, saves them to the out_file, and then calls the evaluation script on it
        """
        pass

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
            "start_logits": output.start_logits,
            "end_logits": output.end_logits,
            "start_index": answer_start_index,
            "end_index": answer_end_index,
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
