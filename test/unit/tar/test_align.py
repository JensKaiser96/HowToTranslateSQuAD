from src.tar.align import Aligner
from src.tar.utils import Span


def test_surface_token_mapping():
    text = (
        "Madam President, I would like to confine my remarks to Alzheimer's "
        "disease.")
    tokens = [
        'Mada', 'm', 'President', ',', 'I', 'would', 'like', 'to', 'confi',
        'ne', 'my', 're', 'marks', 'to', 'Alzheimer', "'", 's', 'disease', '.']

    gold_mapping = {
        (0, "Mada"): Span(0, 4),
        (1, "m"): Span(4, 5),
        (2, "President"): Span(6, 15),
        (3, ","): Span(15, 16),
        (4, "I"): Span(17, 18),
        (5, "would"): Span(19, 24),
        (6, "like"): Span(25, 29),
        (7, "to"): Span(30, 32),
        (8, "confi"): Span(33, 38),
        (9, "ne"): Span(38, 40),
        (10, "my"): Span(41, 43),
        (11, "re"): Span(7, 9),
        (12, "marks"): Span(46, 51),
        (13, "to"): Span(30, 32),
        (14, "Alzheimer"): Span(55, 64),
        (15, "'"): Span(64, 65),
        (16, "s"): Span(9, 10),
        (17, "disease"): Span(67, 74),
        (18, "."): Span(74, 75)
    }

    mapping = Aligner.surface_token_mapping(text, tokens)

    assert gold_mapping == mapping
