from transformers import XLMRobertaTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from typing import Sequence
import string


from src.io.filepaths import Alignment
from src.nlp_tools.span import Span


class Tokenizer:
    """
    Tokenizer used by the Aligner
    """

    # load tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained(
        Alignment.model_path)

    @classmethod
    def encode(cls, source: str, target: str) -> BatchEncoding:
        return cls.tokenizer(source, target, return_tensors="pt")

    @classmethod
    def decode(cls, tokens_ids: Sequence) -> list[str]:
        return [cls.tokenizer.decode(token_id) for token_id in tokens_ids]

    @classmethod
    def split_encoding(cls, encoding: BatchEncoding) -> tuple[Span, Span]:
        """
        extracts spans from an encoding of two texts, the span start index
        is inclusive and the span end is exclusive, e.g.:
            Span(2,5) includes the elements 2, 3, and 4 (not 5)
        It is expected that the encoding has the following format in its
        input_ids tensor:
            [[BOS, <source_text>, EOS, EOS, <target_text>, EOS]]
        """
        BOS = cls.tokenizer.bos_token_id
        EOS = cls.tokenizer.eos_token_id

        ids = list(encoding.input_ids.flatten())

        # check if sequence is as expected
        if not ids[0] == BOS:
            raise ValueError(
                f"Expected sequence to start with [BOS] token (id:{BOS}), "
                f"but sequence starts with id:{ids[0]}.")
        if not ids[-1] == EOS:
            raise ValueError(
                f"Expected sequence to end with [EOS] token (id:{EOS}), "
                f"but sequence ends with id:{ids[-1]}.")
        if not list(ids).count(EOS) == 3:
            raise ValueError(
                f"Expected sequence to have exactly three occurences of the "
                f"[EOS] token (id:{EOS}), but counted {list(ids).count(EOS)} "
                f"instead.")

        first_EOS = ids.index(EOS)

        if not ids[first_EOS + 1] == EOS:
            raise ValueError(
                f"Expected sequence to have the second [EOS] directly follow "
                f" the first [EOS] (id:{EOS}), but the token after the first "
                f"[EOS] has id: {ids[first_EOS + 1]} instead.")

        source = Span(1, first_EOS)
        target = Span(first_EOS + 2, len(ids)-1)

        # Verify spans are not empty
        if source.is_empty:
            raise ValueError(
                f"Source span is not allowed to be empty. {source}")
        if target.is_empty:
            raise ValueError(
                f"Target span is not allowed to be empty. {target}")

        return source, target

    @classmethod
    def surface_token_mapping(cls, text: str, tokens: list[str]) -> list[Span]:
        """
        returns a list of spans corresponding to the tokens in tokens.
        """
        mapping = []
        curser_pos = 0
        for token in tokens:
            # advance curser if the next char is a whitespace.
            while text[curser_pos: curser_pos + 1] in string.whitespace:
                curser_pos += 1
            # create span over current token
            span = Span(curser_pos, curser_pos + len(token))
            # check if content of the span matches with the token
            if token == span(text):
                mapping.append(span)
            else:
                raise ValueError(
                    f"Expected token '{token}' to be at {span.start}: "
                    f"{span.end} in \n'{text}'\n, was: \n'{span(text)}'")
            # move curser to the end of the span
            curser_pos = span.end
        return mapping
