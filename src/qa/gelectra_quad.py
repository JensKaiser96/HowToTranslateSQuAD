import torch
from transformers.tokenization_utils_base import BatchEncoding
from transformers.models.electra.tokenization_electra_fast import (
    ElectraTokenizerFast)
from transformers.models.electra.modeling_electra import (
    ElectraForQuestionAnswering)


from src.nlp_tools.token import Tokenizer, surface_token_mapping
from src.nlp_tools.span import Span


class Gelectra:
    _name = "deepset/gelectra-large-germanquad"
    tokenizer = Tokenizer(ElectraTokenizerFast.from_pretrained(_name))
    model = ElectraForQuestionAnswering.from_pretrained(_name)

    @classmethod
    def prompt(cls, context: str, question: str):
        model_input = cls.tokenizer.encode(context, question)
        with torch.no_grad():
            output = cls.model(**model_input)

        # get answer on token level
        answer_start_index = output.start_logits.argmax()
        answer_end_index = output.end_logits.argmax()
 
        # convert token to surface level
        context_token_ids, _ = cls.split_encoding(model_input)
        context_tokens = cls.tokenizer.decode(context_token_ids(model_input))
        mapping = surface_token_mapping(context, context_tokens, "#")
        return Span.combine(mapping[answer_start_index: answer_end_index + 1])

    @classmethod
    def split_encoding(cls, encoding: BatchEncoding) -> tuple[Span, Span]:
        """
        splits the combined encoding of the ElectraTokenizer of a context
        question pair into its two spans
            [[CLS, <context>, SEP, <question>, SEP]]
        """
        CLS = cls.tokenizer.t.cls_token_id
        SEP = cls.tokenizer.t.sep_token_id

        ids = list(encoding.input_ids.flatten())

        # check if sequence is as expected
        if ids[0] != CLS:
            raise ValueError(
                f"Expected sequence to start with [CLS] token (id:{CLS}), "
                f"but sequence starts with id:{ids[0]}.")
        if ids[-1] != SEP:
            raise ValueError(
                f"Expected sequence to end with [SEP] token (id:{SEP}), "
                f"but sequence ends with id:{ids[-1]}.")
        if list(ids).count(SEP) != 2:
            raise ValueError(
                f"Expected sequence to have exactly three occurences of the "
                f"[SEP] token (id:{SEP}), but counted {list(ids).count(SEP)} "
                f"instead.")

        first_pad = ids.index(SEP)

        context = Span(1, first_pad)
        question = Span(first_pad + 1, len(ids)-1)

        # Verify spans are not empty
        if context.is_empty:
            raise ValueError(
                f"Source span is not allowed to be empty. {context}")
        if question.is_empty:
            raise ValueError(
                f"Target span is not allowed to be empty. {question}")

        return context, question
