from pathlib import Path

from src.io.filepaths import PLOTS_PATH
from src.math.arithmetic import log10_0
from src.nlp_tools.token import get_token_count
from src.plot import scatter, histogram
from src.qa.dataset import Dataset
from src.qa.evaluate import Evaluation, Result
from src.qa.qamodel import QAModel


def plot_scatter(results: Evaluation, name: str, save_path: Path):
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


def plot_hist(results: Evaluation, name: str, save_path: Path):
    metrics = [
        name for name, _type in Result.__annotations__.items() if _type in (float, int)
    ]
    for metric in metrics:
        data = [getattr(result, metric) for result in results.individual_results]
        histogram(data, metric, "Frequency", save_path, title=name)


def plot_hist_comp(results_raw, results_sota, name: str, save_path: Path):
    metrics = [
        name for name, _type in Result.__annotations__.items() if _type in (float, int)
    ]
    for metric in metrics:
        data_raw = [
            getattr(result, metric) for result in results_raw.individual_results
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
            label="raw",
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
    dataset: Dataset = Dataset.GermanQUAD.TEST
    QAModel.lazy_loading = True
    for model in QAModel.get_lazy_qa_instances():
        if model.name == QAModel.EnglishQA.name or model.name == QAModel.RawClean4.name:
            continue
        results = model.get_evaluation(dataset)

        plot_hist(
            results=results,
            name=f"{model.name} {dataset.name}",
            save_path=Path(PLOTS_PATH) / model.name / dataset.name,
        )

        plot_scatter(
            results=results,
            name=f"{model.name} {dataset.name}",
            save_path=Path(PLOTS_PATH) / model.name / dataset.name,
        )

        if model.name != QAModel.GermanQuad.name:
            plot_hist_comp(
                results_raw=model.get_evaluation(dataset),
                results_sota=QAModel.GermanQuad.get_evaluation(dataset),
                name=f"{model.name} vs sota - {dataset.name}",
                save_path=Path(PLOTS_PATH) / f"{model.name}vsSOTA" / dataset.name,
            )
