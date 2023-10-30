import matplotlib.pyplot as plt
import numpy as np

from src.io.utils import str_to_safe_path
from src.misc.colors import RED, YELLOW, GREEN, BLUE, PURPLE, GRAY

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
        kwargs["marker"] = "o"
    if "alpha" not in kwargs:
        kwargs["alpha"] = 0.1
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

    # plt.title(title)
    plt.tight_layout()

    if save_path:
        plot_path = str_to_safe_path(
            save_path / f"scatter_{xlabel}_{ylabel}_{title}.png", replace=True, verbose=False
        )
        plt.savefig(plot_path, dpi=DPI, transparent=True)
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
    # plt.title(title)
    plt.tight_layout()

    if grid:
        plt.grid(axis="y", linestyle="--", alpha=0.7)
    if legend:
        plt.legend()
    if save_path:
        plot_path = str_to_safe_path(
            save_path / f"hist_{xlabel}_{title}", ".png", replace=True, verbose=False
        )
        plt.savefig(plot_path, dpi=DPI, transparent=True)
        plt.close()


def plot_4bars(keys, squad_values, raw_values, tar_values, quote_values, save_path):
    # Define bar width and positions
    bar_width = 0.2
    bar_positions = np.arange(len(keys))

    # Create the bar graph
    plt.bar(bar_positions - 1.5 * bar_width, squad_values, width=bar_width, label='squad', color=RED)
    plt.bar(bar_positions - 0.5 * bar_width, raw_values, width=bar_width, label='raw', color=YELLOW)
    plt.bar(bar_positions + 0.5 * bar_width, tar_values, width=bar_width, label='tar', color=GREEN)
    plt.bar(bar_positions + 1.5 * bar_width, quote_values, width=bar_width, label='quote', color=BLUE)

    plt.ylabel('%')
    plt.xticks(bar_positions, keys)
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, transparent=True)


def plot_3bars(keys, raw_values, tar_values, quote_values, save_path):
    # Define bar width and positions
    bar_width = 0.25
    bar_positions = np.arange(len(keys))

    # Create the bar graph
    plt.bar(bar_positions - 1 * bar_width, raw_values, width=bar_width, label='raw', color=YELLOW)
    plt.bar(bar_positions + 0 * bar_width, tar_values, width=bar_width, label='tar', color=GREEN)
    plt.bar(bar_positions + 1 * bar_width, quote_values, width=bar_width, label='quote', color=BLUE)

    plt.ylabel('% correct')
    plt.xticks(bar_positions, keys)
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, transparent=True)


def plot_51bars(keys, values, big_values, save_path):
    # Define bar width and positions
    bar_width = 1/6
    bar_positions = np.arange(len(keys))

    # Create the bar graph
    plt.bar(bar_positions - 1 * bar_width, values["date"], width=bar_width, label='date', color=RED)
    plt.bar(bar_positions - 0.5 * bar_width, values["number"], width=bar_width, label='number', color=YELLOW)
    plt.bar(bar_positions + 0 * bar_width, values["capital"], width=bar_width, label='capital', color=GREEN)
    plt.bar(bar_positions + 0.5 * bar_width, values["lower"], width=bar_width, label='lower', color=BLUE)
    plt.bar(bar_positions + 1 * bar_width, values["none"], width=bar_width, label='none', color=PURPLE)

    plt.bar(bar_positions, big_values, width=1, alpha=0.3, color=GRAY)

    plt.ylabel('% answer type / % correct')
    plt.xlabel('gold answer types')
    plt.xticks(bar_positions, keys)
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, transparent=True)
