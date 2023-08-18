from src.qa.gelectra import Gelectra
from src.qa.quad import QUAD
from src.utils.logging import get_logger

logger = get_logger(__file__)


def main():
    raw_clean_model = Gelectra.RawClean
    raw_clean_model.evaluate(QUAD.GermanQUAD.TEST)


if __name__ == "__main__":
    main()
