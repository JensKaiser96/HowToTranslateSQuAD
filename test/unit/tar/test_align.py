from src.nlp_tools.span import Span
from src.nlp_tools.token import surface_token_mapping
from src.utils.logging import get_logger

logger = get_logger(__name__)


def test_surface_token_mapping():
    text = (
        "Madam President, I would like to confine my remarks to Alzheimer's "
        "disease ."
    )
    tokens = [
        "Mada",
        "m",
        "President",
        ",",
        "I",
        "would",
        "like",
        "to",
        "confi",
        "ne",
        "my",
        "re",
        "marks",
        "to",
        "Alzheimer",
        "'",
        "s",
        "disease",
        "",
        ".",
    ]

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
        (11, "re"): Span(44, 46),
        (12, "marks"): Span(46, 51),
        (13, "to"): Span(52, 54),
        (14, "Alzheimer"): Span(55, 64),
        (15, "'"): Span(64, 65),
        (16, "s"): Span(65, 66),
        (17, "disease"): Span(67, 74),
        (18, ""): Span(75, 75),
        (19, "."): Span(75, 76),
    }

    mapping = surface_token_mapping(text, tokens)

    for (index, token), gold_span in gold_mapping.items():
        predicted_span = mapping[index]
        logger.info(f"({index}){token}, Span={predicted_span}")
        assert gold_span == predicted_span
