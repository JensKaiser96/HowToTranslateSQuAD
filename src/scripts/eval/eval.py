from src.qa.gelectra import Gelectra
from src.qa.quad import QUAD
from src.utils.logging import get_logger

logger = get_logger(__file__, script=True)


def main():
    # Model
    model_names = ["RawClean", "GermanQUAD"]
    print("Chose which model you would like to evaluate:")
    for i, model_name in enumerate(model_names):
        print(f"\t[{i}] - {model_name}")
    chosen_model_index = int(input("Enter Number associated with the model: "))
    chosen_model_name = model_names[chosen_model_index]
    print(f"Chosen Model: {chosen_model_name}")

    # Dataset
    dataset_names = [
        "GermanQUAD.TEST",
        "StressTest.NOT",
        "StressTest.DIS",
        "StressTest.ONE",
        "StressTest.ODD",
    ]
    print("Chose which dataset you want to evaluate the model on:")
    for i, dataset_name in enumerate(dataset_names):
        print(f"\t[{i}] - {dataset_name}")
    chosen_dataset_index = int(input("Enter Number associated with the dataset: "))
    print(f"Chosen Dataset: {dataset_names[chosen_dataset_index]}")
    dataset_parent_name, dataset_child_name = dataset_names[chosen_dataset_index].split(
        "."
    )
    dataset_parent = getattr(QUAD, dataset_parent_name)
    dataset = getattr(dataset_parent, dataset_child_name)

    if Gelectra.has_results_file(chosen_model_name, dataset.name):
        evaluate_again = input(
            f"{chosen_model_name} has already been evaluated on {dataset.name}.\n"
            f"Do you want to do it again (y/N)"
        )
        if evaluate_again.lower() != "y":
            print("Exiting ...")
            return

    model = getattr(Gelectra, model_names[chosen_model_index])
    model.evaluate(dataset, f"{model.name}")


if __name__ == "__main__":
    main()
