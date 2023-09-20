import datetime
import json
import os
from pathlib import Path
from typing import Union

from src.io.filepaths import PLOTS_PATH


def save_plt(plt, path: str):
    suffix = ".png"
    if "/plots" not in path:
        path = PLOTS_PATH + path
    path = str_to_safe_path(path, suffix)
    plt.savefig(path, transparent=True)


def to_json(data: Union[dict, str], path: str):
    suffix = ".json"
    path = str_to_safe_path(path, suffix)
    with open(path, "w", encoding="utf-8") as f_out:
        if isinstance(data, dict):
            json.dump(data, f_out, ensure_ascii=False, indent=4)
        else:
            f_out.write(data)


def _fix_relative_paths(path: str) -> str:
    # is not absolute and not specifically relative
    if path[0] != "/" and path[0] != ".":
        path = "./" + path  # make relative
    return path


def _rename_old_file(path: Path, verbose=False):
    file_name, file_ext = os.path.splitext(path)
    modified_date = datetime.datetime.fromtimestamp(os.path.getmtime(path))
    modified_date_str = modified_date.strftime("%Y_%m_%d_%H_%M_%S")
    new_file_name = f"{file_name}_{modified_date_str}{file_ext}"
    if verbose:
        print(f"file at given path {path} already exists, renaming old file")
    os.rename(path, new_file_name)


def str_to_safe_path(filepath: str, suffix: str = "", verbose=False):
    fixed_path = _fix_relative_paths(filepath)
    path = Path(fixed_path)
    if (
        suffix and path.suffix != suffix
    ):  # set the suffix if explicitly given and not already set via path
        path = path.with_suffix(suffix)
    # warn if suffix is given neiter explicitly nor implicitly
    elif not path.suffix and not path.is_dir():
        print("No suffix in filepath or in the suffix argument provided.")

    path.parent.mkdir(exist_ok=True, parents=True)  # create parent dir

    if os.path.exists(path):
        _rename_old_file(path, verbose)
    return path
