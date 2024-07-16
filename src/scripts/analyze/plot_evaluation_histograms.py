from pathlib import Path

from huggingface_hub.utils import HFValidationError

from src.io.filepaths import PLOTS, Models, Datasets
from src.math.arithmetic import log10_0
from src.nlp_tools.token import get_token_count
from src.plot import scatter, histogram
from src.qa.dataset import Dataset
from src.qa.evaluate_predictions import PredictionEvaluation, Result
from src.qa.qamodel import QAModel


def plot_scatter(results: PredictionEvaluation, name: str, save_path: Path):
    # confidence, f1
    # recall vs precision
    # confidence start, confidence end
    confidence = [
        result.confidence_start * result.confidence_end
        for result in results.individual_results
    ]
    f1 = [result.F1 for result in results.individual_results]
    recall = [result.recall for result in results.individual_results]
    precision = [result.precision for result in results.individual_results]
    start = [result.confidence_start for result in results.individual_results]
    end = [result.confidence_end for result in results.individual_results]
    predicted_answer_length = [log10_0(get_token_count(result.model_output.text)) for result in results.individual_results]
    gold_answer_length = [log10_0(get_token_count(result.best_answer)) for result in results.individual_results]

    comparisions = [
        ["confidence", confidence, "F1", f1],
        ["recall", recall, "precision", precision],
        ["confidence_start", start, "confidence_end", end],
        ["predicted_answer_length", predicted_answer_length, "gold_answer_length", gold_answer_length],
    ]
    for xlabel, xdata, ylabel, ydata in comparisions:
        scatter(xlabel, xdata, ylabel, ydata, save_path, name, new_fig=True)


def plot_hist(results: PredictionEvaluation, name: str, save_path: Path):
    metrics = [
        name for name, _type in Result.__annotations__.items() if _type in (float, int)
    ]
    for metric in metrics:
        data = [getattr(result, metric) for result in results.individual_results]
        histogram(data, metric, "Frequency", save_path, title=name)


def plot_hist_comp(results, label, results_sota, name: str, save_path: Path):
    metrics = [
        name for name, _type in Result.__annotations__.items() if _type in (float, int)
    ]
    for metric in metrics:
        data_raw = [
            getattr(result, metric) for result in results.individual_results
        ]
        data_sota = [
            getattr(result, metric) for result in results_sota.individual_results
        ]
        histogram(
            data=data_raw,
            xlabel=metric,
            ylabel="Frequency",
            new_fig=True,
            save_path="",
            title=name,
            alpha=0.5,
            color="blue",
            label=label,
        )
        histogram(
            data=data_sota,
            xlabel=metric,
            ylabel="Frequency",
            new_fig=False,
            save_path=save_path,
            title=name,
            alpha=0.5,
            color="red",
            label="sota",
            legend=True,
        )


if __name__ == '__main__':
    dataset = Dataset.load(Datasets.GermanQuad.DEV)
    QAModel.lazy_loading = True
    for model in QAModel.get_all_models():
        if model.path == Models.QA.Distilbert.ENGLISH_QA or model.path == Models.QA.Gelectra.RAW_CLEAN_4:
            continue
        try:
            results = model.get_evaluation(dataset)
        except HFValidationError:
            print(f"Loading Evaluations failed for {model.name}, skipping ...")
            continue

        plot_hist(
            results=results,
            name=f"{model.name} {dataset.name}",
            save_path=Path(PLOTS) / model.name / dataset.name,
        )

        plot_scatter(
            results=results,
            name=f"{model.name} {dataset.name}",
            save_path=Path(PLOTS) / model.name / dataset.name,
        )

        if model.path != Models.QA.Gelectra.GERMAN_QUAD:
            plot_hist_comp(
                results=model.get_evaluation(dataset),
                label=model.name.split(".")[-1],
                results_sota=QAModel(Models.QA.Gelectra.GERMAN_QUAD).get_evaluation(dataset),
                name=f"{model.name} vs sota - {dataset.name}",
                save_path=Path(PLOTS) / f"{model.name}vsSOTA" / dataset.name,
            )
