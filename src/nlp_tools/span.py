from dataclasses import dataclass
from typing import Sequence, Union

import torch
from transformers.tokenization_utils_base import BatchEncoding

from src.qa.dataset import Answer


@dataclass
class Span:
    start: int
    end: int

    def __init__(self, start: int, end: int, relative=False):
        self.start = start
        if relative:
            self.end = start + end
        else:
            self.end = end

    @classmethod
    def from_answer(cls, answer: Answer):
        start = answer.answer_start
        end = start + len(answer.text)
        return cls(start, end)

    @classmethod
    def combine(cls, spans: list["Span"]):
        if not spans:
            return cls(0, 0)
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
        return sequence[self.start : self.end]

    def __len__(self) -> int:
        return self.end - self.start

    def __add__(self, other: "Span") -> "Span":
        return Span(start=min(self.start, other.start), end=max(self.end, other.end))

    def compare(self, other: "Span") -> int:
        """
        returns the sum of the difference of start and end
        """
        return abs(self.start - other.start) + abs(self.end - other.end)

    @property
    def is_empty(self) -> bool:
        return self.start >= self.end

    def is_subspan(self, other: "Span") -> bool:
        return self.start >= other.start and self.end <= other.end
