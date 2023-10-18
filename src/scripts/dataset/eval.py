import sys

from src.qa.dataset import Dataset

if __name__ == '__main__':
    if len(sys.argv) == 2:
        fuzzy_dataset_name = sys.argv[1]
        dataset = Dataset.from_fuzzy(fuzzy_dataset_name)
        dataset.get_evaluation()
    else:
        raise ValueError("Please specify the dataset you want to evaluate.")
