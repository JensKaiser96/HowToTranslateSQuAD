import tqdm
from src.tools.logging import get_logger
from src.tools.formatter import Formatter
from src.tools.quad import QUAD
logger = get_logger(__file__, script=True)

"""
Script to help annotate unanswerable questions.
The script loops over the all contexts, in the base dataset and shows it
to the user alongside the questions. The user is then able to input a new
unanswerable question. All new questions are saved to the annotated dataset.
"""


def main():
    f = Formatter()
    not_base = QUAD(QUAD.StressTest.Base.NOT)
    not_annotated = QUAD(QUAD.StressTest.NOT)
    for entry in tqdm.tqdm(not_base.data):
        for paragraph in entry:
            context = paragraph.context
            f.print(f"\n\n{context}\n")
            for qa in paragraph.qas:
                f.print(f"\t- {qa.question}")
            user_input = input(
                    "\n\nEnter unanswerable Question (leave blank to skip):\n")
            if not user_input:
                continue
            not_annotated.add_unanswerable_question(context, user_input)
            # Save every time, this will cause a lot of old files, but I can live with that
            not_annotated.save(QUAD.StressTest.NOT, version="NOT_annotated")


if __name__ == "__main__":
    main()