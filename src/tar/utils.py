"""
Code is taken from:
https://github.com/CZWin32768/XLM-Align/blob/main/word_aligner/xlmalign-ot-aligner.py
with some modifications for readability
"""
from dataclasses import dataclass
import torch
from typing import Sequence, Union
from transformers.tokenization_utils_base import BatchEncoding
from enum import Enum, auto

from src.qa.quad import Answer
from src.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    @classmethod
    def from_answer(cls, answer: Answer):
        start = answer.answer_start
        end = start + len(answer.text)
        return cls(start, end)

    @classmethod
    def combine(cls, spans: list["Span"]):
        min_start = min(spans, key=lambda span: span.start)
        max_end = max(spans, key=lambda span: span.end)
        return cls(min_start.start, max_end.end)

    def __call__(self, sequence: Union[Sequence, BatchEncoding]) -> Sequence:
        # special case if the given sequence is a BatchEncoding, plus if the
        # input_ids are Tensors
        if isinstance(sequence, BatchEncoding):
            sequence = sequence.input_ids
            if isinstance(sequence, torch.Tensor) and sequence.dim() > 1:
                sequence = sequence.flatten()
        return sequence[self.start: self.end]

    def __len__(self) -> int:
        return self.end - self.end

    def __add__(self, other: "Span") -> "Span":
        return Span(
            start=min(self.start, other.start),
            end=max(self.end, other.end))

    @property
    def is_empty(self) -> bool:
        return self.start >= self.end

    def is_subspan(self, other: "Span") -> bool:
        return self.start >= other.start and self.end <= other.end


class Direction(Enum):
    forwards = auto()
    backwards = auto()
    bidirectional = auto()
