import string
from typing import Sequence

from transformers.tokenization_utils_base import BatchEncoding

from src.nlp_tools.span import Span


class Tokenizer:
    def __init__(self, tokenizer):
        self.model = tokenizer
        self.max_length = tokenizer.max_len_single_sentence

    def encode_align(self, *text: str) -> BatchEncoding:
        # QA: first question then context
        return self.model(
            *text,
            return_tensors="pt",
        )

    def encode_qa(self, question: str, context: str):
        return self.model(
            question,
            context,
            return_tensors="pt",
            truncation="only_second",
            max_length=self.max_length,
            stride=self.max_length // 3,  # overlap 1/3 of total length
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

    def decode(self, tokens_ids: Sequence) -> list[str]:
        return [self.model.decode(token_id) for token_id in tokens_ids]


def blank_or_weird_char(text: str, curser_pos: int) -> bool:
    char = text[curser_pos : curser_pos + 1]
    return char in string.whitespace or char not in string.printable


def surface_token_mapping(
    text: str, tokens: list[str], padding_char: str = ""
) -> list[Span]:
    """
    encode(..., return_offsets_mapping=True).offset_mapping[[token0.start, token0.end], [token1.start, token1.end], ...]
    returns a list of spans corresponding to the tokens in tokens.
    """
    mapping = []
    curser_pos = 0
    last_unknown = False
    for i, token in enumerate(tokens):
        # remove padding but don't remove it if its just that char
        if padding_char and token != padding_char:
            token = token.strip(padding_char)
        # advance curser if the next char a whitespace
        while text[curser_pos] in string.whitespace:
            curser_pos += 1
            # break loop if end of text is reached. Not sure if that ever happens before the last token is reached
            if curser_pos >= len(text):
                raise ValueError(
                    f"Reached end of text before consuming all tokens. Remaining tokens:\n{tokens[i:]}"
                )
        # check if the next char is a weired symbol, i.e. not in [a-Z0-9.-`]
        weird_char = any([char not in string.printable for char in text[curser_pos: curser_pos + len(token)]])

        # create span over current token and deal with last unknown token
        if last_unknown:
            # search for current token after the curser_pos
            span = Span(text.find(token, curser_pos), len(token), relative=True)
            last_span = Span(curser_pos, span.start)
            mapping.append(last_span)
        else:
            span = Span(curser_pos, len(token), relative=True)

        # check if content of the span matches with the token
        if token == span(text):
            mapping.append(span)
        elif token == "[UNK]" or weird_char or len(token) == 1:
            last_unknown = True
            continue
        else:
            raise ValueError(
                f"Expected token '{token}' to be at {span.start}: {span.end} in \n"
                f"...{text[span.start - 100: span.end + 100]}...\n"
                f"was: '{span(text)}'\n"
                f"tokens: ... {tokens[i - 3:i + 3]} ...\n"
                f"mapping: {mapping}"
            )
        # move curser to the end of the span
        curser_pos = span.end
    return mapping
