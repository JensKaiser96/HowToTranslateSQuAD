from src.tar.translate import Translator


def test_translator():
    t = Translator()
    t.de2en("Test")
    t.en2de("Test")
