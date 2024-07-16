import datetime
import json
import os
from pathlib import Path
from typing import Union

from src.io.filepaths import Paths


def save_plt(plt, path: str):
    suffix = ".pdf"
    if "/plots" not in path:
        path = Paths.PLOTS / path
    path = make_path_safe(path, suffix)
    plt.savefig(path, transparent=True)


def to_json(data: Union[dict, str], path: str):
    suffix = ".json"
    path = make_path_safe(path, suffix)
    with open(path, "w", encoding="utf-8") as f_out:
        if isinstance(data, dict):
            json.dump(data, f_out, ensure_ascii=False, indent=4)
        else:
            f_out.write(data)


def make_path_safe(
    filepath: Union[str, Path], suffix: str = "", verbose=True, replace=False, dir_ok=False
):
    if isinstance(filepath, Path):
        filepath = str(filepath)
    fixed_path = _fix_relative_paths(filepath)
    fixed_path = fixed_path.replace(" ", "_")  # replace space with '_'
    path = Path(fixed_path)
    if suffix:  # set the suffix if explicitly given
        path = path.with_suffix(suffix)
    # warn if suffix is given neither explicitly nor implicitly
    elif not path.suffix and not path.is_dir():
        print("No suffix in filepath or in the suffix argument provided.")

    path.parent.mkdir(exist_ok=True, parents=True)  # create parent dir

    if os.path.exists(path):
        if path.is_dir() and dir_ok:
            pass  # dont create new dir
        elif replace:  # do nothing, maybe say what you replace
            if verbose:
                print(f"src.io.utils.str_to_safe_path.py [I]:\nREPLACING '{path}' with new file")
        else:
            _rename_old_file(path)
    return path


def _rename_old_file(path: Path, verbose=False):
    file_name, file_ext = os.path.splitext(path)
    modified_date = datetime.datetime.fromtimestamp(os.path.getmtime(path))
    modified_date_str = modified_date.strftime("%Y_%m_%d_%H_%M_%S")
    new_file_name = f"{file_name}_{modified_date_str}{file_ext}"
    if verbose:
        print(f"file at given path {path} already exists, renaming old file")
    os.rename(path, new_file_name)


def _fix_relative_paths(path: str) -> str:
    # is not absolute and not specifically relative
    if path[0] != "/" and path[0] != ".":
        path = "./" + path  # make relative
    return path
