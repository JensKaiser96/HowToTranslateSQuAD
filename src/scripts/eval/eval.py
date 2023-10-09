import sys

from src.nlp_tools.fuzzy import fuzzy_match
from src.qa.dataset import Dataset
from src.qa.qamodel import QAModel
from src.utils.logging import get_logger

logger = get_logger(__file__, script=True)


def main(fuzzy_model_name, fuzzy_dataset_name):
    model_names = QAModel.get_model_names()
    dataset_names = Dataset.get_dataset_names()

    chosen_model_name = fuzzy_match(fuzzy_model_name, model_names)
    chosen_dataset_name = fuzzy_match(fuzzy_dataset_name, dataset_names)

    if chosen_model_name is None:
        print("Chose which model you would like to evaluate:")
        for i, model_name in enumerate(model_names):
            print(f"\t[{i}] - {model_name}")
        chosen_model_index = int(input("Enter Number associated with the model: "))
        chosen_model_name = model_names[chosen_model_index]
    print(f"Chosen Model: {chosen_model_name}")

    if chosen_dataset_name is None:
        print("Chose which dataset you want to evaluate the model on:")
        for i, dataset_name in enumerate(dataset_names):
            print(f"\t[{i}] - {dataset_name}")
        chosen_dataset_index = int(input("Enter Number associated with the dataset: "))
        chosen_dataset_name = dataset_names[chosen_dataset_index]
    print(f"Chosen Dataset: {chosen_dataset_name}")

    # load model + dataset
    QAModel.lazy_loading = True  # only load model weights once needed
    model: QAModel = getattr(QAModel, chosen_model_name)
    dataset_parent_name, dataset_child_name = chosen_dataset_name.split(".")
    dataset_parent = getattr(Dataset, dataset_parent_name)
    dataset = getattr(dataset_parent, dataset_child_name)

    if model.has_results_file(dataset.name):
        print("This model already has a results file, creating a new one unless you stop me :|")

    model.get_evaluation(dataset, redo=True)


if __name__ == "__main__":
    model_name = ""
    dataset_name = ""
    if len(sys.argv) == 3:
        model_name = sys.argv[1]
        dataset_name = sys.argv[2]

    main(model_name, dataset_name)
