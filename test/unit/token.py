from src.nlp_tools.span import Span
from src.nlp_tools.token import surface_token_mapping


def test_token_surface_mapping():
    text = "2001, about 8½ months after"
    tokens = ['2001,', 'about', '8', '1⁄2', 'months', 'after']
    mapping = surface_token_mapping(text, tokens)
    assert mapping == [Span(start=0, end=5), Span(start=6, end=11), Span(start=12, end=13), Span(start=13, end=15), Span(start=15, end=21), Span(start=22, end=22), Span(start=22, end=27)]

    text = "say pin [pʰɪn]"
    tokens = ["say", "pin", "[", "ph", "ɪ", "n"]
    mapping = surface_token_mapping(text, tokens)
    assert mapping == [
        Span(start=0, end=3),
        Span(start=4, end=7),
        Span(start=8, end=9),
        Span(start=10, end=12),
        Span(start=13, end=14),
        Span(start=22, end=22),
        Span(start=22, end=27),
    ]
