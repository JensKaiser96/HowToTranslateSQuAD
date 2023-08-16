from src.qa.gelectra import Gelectra
from src.qa.quad import QUAD


def main():
    raw_clean_model = Gelectra.RawClean
    raw_clean_model.evaluate(QUAD.GermanQUAD.TEST, "test")


if __name__ == '__main__':
    main()