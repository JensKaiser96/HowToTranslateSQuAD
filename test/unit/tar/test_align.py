from src.tar.align import Aligner


def test_alinger():
    aliner = Aligner()
    result = aliner("This is a test.", "Dies ist ein Test.")
    print(result)
