import string
from typing import Sequence

from transformers.tokenization_utils_base import BatchEncoding

from src.nlp_tools.span import Span


class Tokenizer:
    def __init__(self, tokenizer):
        self.model = tokenizer
        self.max_length = tokenizer.max_len_single_sentence

    def encode(self, *text: str) -> BatchEncoding:
        # QA: first question then context
        return self.model(
            *text,
            return_tensors="pt",
            truncation="only_second",
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
    last_unknown = False
    for i, token in enumerate(tokens):
        if (
            padding_char and token != padding_char
        ):  # remove padding but don't remove it if its just that char
            token = token.strip(padding_char)
        # advance curser if the next char is a whitespace, or ASCII CHAR 160 (non-breaking space), or S
        while (
            text[curser_pos : curser_pos + 1] in string.whitespace  # skip if whitespace
            or ord(text[curser_pos : curser_pos + 1]) == 160  # non-breaking space
            or ord(text[curser_pos : curser_pos + 1]) == 173  # soft-hyphen
        ):
            curser_pos += 1

        # create span over current token and deal with last unknown token
        if last_unknown:
            span = Span(text.find(token, curser_pos), len(token), absolute=False)
            last_span = Span(curser_pos, span.start)
            mapping.append(last_span)
        else:
            span = Span(curser_pos, curser_pos + len(token))

        # check if content of the span matches with the token
        if token == span(text):
            mapping.append(span)
        elif token == "[UNK]":
            last_unknown = True
            continue
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
