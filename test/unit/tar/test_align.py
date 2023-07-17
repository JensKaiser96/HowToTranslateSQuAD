from src.tar.align import Aligner


def test_extrapolation():
    test_sequence = [(1, 1), (2, 2), (4, 4)]
    gold_sequence = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    assert Aligner.extrapolate_alignment(test_sequence) == gold_sequence
