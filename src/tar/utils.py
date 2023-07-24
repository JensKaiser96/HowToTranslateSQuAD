"""
Code is taken from:
https://github.com/CZWin32768/XLM-Align/blob/main/word_aligner/xlmalign-ot-aligner.py
with some modifications for readability
"""
from dataclasses import dataclass
import torch
from typing import Sequence, Union
from transformers.tokenization_utils_base import BatchEncoding

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
        return cls(min_start, max_end)

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


def _extract_wa_from_sim(sim: torch.Tensor):
    logger.debug/(f"sim:\n{sim}")
    m, n = sim.size()
    forward = torch.eye(n)[sim.argmax(dim=1)]
    backward = torch.eye(m)[sim.argmax(dim=0)]
    inter = forward * backward.transpose(0, 1)
    logger.debug(f"inter:\n{inter}")
    return [(i, j) for i, j in (inter > 0).nonzero()]


def _sinkhorn_iter(S: torch.Tensor, num_iter=2) -> torch.Tensor:
    if num_iter <= 0:
        return S, S
    if not S.dim() == 2:
        raise ValueError(
            f"Expected S.dim() to be 2, but was '{S.dim()}' instead \n{S=}")
    S[S <= 0].fill_(1e-6)
    for _ in range(num_iter):
        S = S / S.sum(dim=0, keepdim=True)
        S = S / S.sum(dim=1, keepdim=True)
    return S


def sinkhorn(sim: torch.Tensor, source: Span, target: Span, num_iter=2
             ) -> list[tuple[int, int]]:
    # check for valid spans
    if source.is_empty:
        logger.warn(f"source span is empty: {source.start=}, {source.end=}")
        return []
    if target.is_empty:
        logger.warn(f"target span is empty: {target.start=}, {target.end=}")
        return []
    sim_wo_offset = sim[source.start: source.end, target.start: target.end]
    sim = _sinkhorn_iter(sim_wo_offset, num_iter)
    pred_wa_wo_offset = _extract_wa_from_sim(sim)
    return [(source_offset + source.start, target_offset + target.start) for
            source_offset, target_offset in pred_wa_wo_offset]


def batch_sinkhorn(batch_sim: torch.Tensor, sources: list[Span],
                   targets: list[Span], num_iter=2):
    predicted_word_alignments = []
    for sim, source, target in zip(batch_sim, sources, targets):
        predicted_word_alignments.append(sinkhorn(sim, source, target))
    return predicted_word_alignments
