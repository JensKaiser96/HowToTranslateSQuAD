import matplotlib.pyplot as plt
import numpy as np

from src.io.utils import str_to_safe_path

bin_size = 0.01
margin_ratio = 0.025
DPI = 300


def scatter(
    xlabel,
    xdata,
    ylabel,
    ydata,
    save_path="",
    title="",
    new_fig=True,
    limits=None,
    grid=True,
    **kwargs,
):
    # default args
    if "marker" not in kwargs:
        kwargs["marker"] = "x"
    if limits is None:
        limits = (
            min(0, min(xdata)),
            max(1, max(xdata)),
            min(0, min(ydata)),
            max(1, max(ydata)),
        )

    if new_fig:
        plt.figure()

    plt.scatter(xdata, ydata, **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    xlim_min, xlim_max, ylim_min, ylim_max = limits
    xmargin = (xlim_max - xlim_min) * margin_ratio
    ymargin = (ylim_max - ylim_min) * margin_ratio
    plt.xlim(xlim_min - xmargin, xlim_max + xmargin)
    plt.ylim(ylim_min - ymargin, ylim_max + ymargin)

    if grid:
        plt.grid(True, linestyle="--", alpha=0.5)

    plt.title(title)

    if save_path:
        plot_path = str_to_safe_path(
            save_path / f"scatter_{xlabel}_{ylabel}_{title}.png", replace=True
        )
        plt.savefig(plot_path, dpi=DPI)
        plt.close()


def histogram(
    data,
    xlabel,
    ylabel,
    save_path="",
    new_fig=True,
    title="",
    bins=None,
    limits=None,
    grid=True,
    legend=False,
    **kwargs,
):
    if "edgecolor" not in kwargs:
        kwargs["edgecolor"] = "black"
    if bins is None:
        bins = np.arange(
            0, 1 + bin_size, bin_size
        )  # Bins from 0 to 1 in 0.1 increments
    if limits is None:
        limits = (0, 1, None, None)

    if new_fig:
        plt.figure()
    plt.hist(data, bins=bins, **kwargs)

    # limits
    xlim_min, xlim_max, ylim_min, ylim_max = limits
    if xlim_min is not None and xlim_max is not None:
        plt.xlim(xlim_min, xlim_max)
    if ylim_min is not None and ylim_max is not None:
        plt.xlim(ylim_min, ylim_max)

    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if grid:
        plt.grid(axis="y", linestyle="--", alpha=0.7)
    if legend:
        plt.legend()
    if save_path:
        plot_path = str_to_safe_path(
            save_path / f"hist_{xlabel}_{title}", ".png", replace=True
        )
        plt.savefig(plot_path, dpi=DPI)
        plt.close()
