import string
from typing import Sequence

from transformers.tokenization_utils_base import BatchEncoding

from src.nlp_tools.span import Span


class Tokenizer:
    def __init__(self, tokenizer):
        self.model = tokenizer

    def encode(self, *text: str) -> BatchEncoding:
        return self.model(*text, return_tensors="pt")

    def decode(self, tokens_ids: Sequence) -> list[str]:
        return [self.model.decode(token_id) for token_id in tokens_ids]


def surface_token_mapping(text: str, tokens: list[str], padding_char: str = ""
                          ) -> list[Span]:
    """
    returns a list of spans corresponding to the tokens in tokens.
    """
    mapping = []
    curser_pos = 0
    for token in tokens:
        if padding_char:
            token = token.strip(padding_char)
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
