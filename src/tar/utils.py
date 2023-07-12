"""
Code is taken from:
https://github.com/CZWin32768/XLM-Align/blob/main/word_aligner/xlmalign-ot-aligner.py
with some modifications for readability
"""
import torch

from src.utils.logging import get_logger


logger = get_logger(__name__)


class Span:
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end

    @property
    def is_empty(self) -> bool:
        return self.start <= self.end


def _extract_wa_from_pi_xi(pi, xi):
    m, n = pi.size()
    forward = torch.eye(n)[pi.argmax(dim=1)]
    backward = torch.eye(m)[xi.argmax(dim=0)]
    inter = forward * backward.transpose(0, 1)
    ret = []
    for i in range(m):
        for j in range(n):
            if inter[i, j].item() > 0:
                ret.append((i, j))
    return ret


def _sinkhorn_iter(S, num_iter=2) -> tuple[int, int]:
    if num_iter <= 0:
        return S, S
    assert S.dim() == 2
    S[S <= 0].fill_(1e-6)
    pi = S
    xi = pi
    for i in range(num_iter):
        pi_sum_over_i = pi.sum(dim=0, keepdim=True)
        xi = pi / pi_sum_over_i
        xi_sum_over_j = xi.sum(dim=1, keepdim=True)
        pi = xi / xi_sum_over_j
    return pi, xi


def sinkhorn(sim: torch.Tensor, source: Span, target: Span, num_iter=2
             ) -> list[tuple[int, int]]:
    # check for valid spans
    if source.is_empty:
        logger.warn(f"source span is empty: {source.start=}, {source.end}")
        return []
    if target.is_empty:
        logger.warn(f"target span is empty: {target.start=}, {target.end}")
        return []

    sim_wo_offset = sim[source.start: source.end, target.start: target.end]
    pi, xi = _sinkhorn_iter(sim_wo_offset, num_iter)
    pred_wa_wo_offset = _extract_wa_from_pi_xi(pi, xi)
    return [(source_offset + source.start, target_offset + target.start) for
            source_offset, target_offset in pred_wa_wo_offset]


def batch_sinkhorn(batch_sim: torch.Tensor, sources: list[Span],
                   targets: list[Span], num_iter=2):
    predicted_word_alignments = []
    for sim, source, target in zip(batch_sim, sources, targets):
        predicted_word_alignments.append(sinkhorn(sim, source, target))
    return predicted_word_alignments
