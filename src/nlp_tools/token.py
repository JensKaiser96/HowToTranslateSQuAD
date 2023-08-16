import string
from typing import Sequence

from transformers.tokenization_utils_base import BatchEncoding

from src.nlp_tools.span import Span


class Tokenizer:
    def __init__(self, tokenizer):
        self.model = tokenizer
        self.max_length = tokenizer.max_len_single_sentence

    def encode(self, *text: str) -> BatchEncoding:
        return self.model(
            *text,
            return_tensors="pt",
            truncation="only_first",
            max_length=self.max_length,
            stride=self.max_length // 3,  # overlap 1/3 of total length
        )

    def decode(self, tokens_ids: Sequence) -> list[str]:
        return [self.model.decode(token_id) for token_id in tokens_ids]


def surface_token_mapping(
    text: str, tokens: list[str], padding_char: str = ""
) -> list[Span]:
    """
    returns a list of spans corresponding to the tokens in tokens.
    """
    mapping = []
    curser_pos = 0
    for token in tokens:
        if padding_char:
            token = token.strip(padding_char)
        # advance curser if the next char is a whitespace, or ASCII CHAR 160 (non-breaking space)
        while (
            text[curser_pos : curser_pos + 1] in string.whitespace
            or ord(text[curser_pos : curser_pos + 1]) == 160
        ):
            curser_pos += 1
        # create span over current token
        span = Span(curser_pos, curser_pos + len(token))
        # check if content of the span matches with the token
        if token == span(text):
            mapping.append(span)
        else:
            raise ValueError(
                f"Expected token '{token}' to be at {span.start}: {span.end} in \n"
                f"...{text[span.start - 100: span.end + 100]}...\n"
                f"was: '{span(text)}'\n"
                f"tokens: ...{tokens[i-3:i+3]}...\n"
                f"mapping: {mapping}"
            )
        # move curser to the end of the span
        curser_pos = span.end
    return mapping
