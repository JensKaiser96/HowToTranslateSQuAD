import datetime
import os
import json
from pathlib import Path


def to_json(data: dict, path: str):
    suffix = ".json"
    path = _str_to_safe_path(path, suffix)
    with open(path, 'w', encoding='utf-8') as f_out:
        json.dump(data, f_out, ensure_ascii=False, indent=4)


def _fix_relative_paths(path: str) -> str:
    # is not absolute and not specifically relative
    if path[0] != "/" and path[0] != ".":
        path = "./" + path  # make relative
    return path


def _rename_old_file(path: str):
    file_name, file_ext = os.path.splitext(path)
    modified_date = datetime.datetime.fromtimestamp(os.path.getmtime(path))
    modified_date_str = modified_date.strftime("%Y_%m_%d_%H_%M_%S")
    new_file_name = f"{file_name}_{modified_date_str}{file_ext}"
    os.rename(path, new_file_name)
    print(f"Renamed '{path}' to '{new_file_name}'")


def _str_to_safe_path(filepath: str, suffix):
    fixed_path = _fix_relative_paths(filepath)
    path = Path(fixed_path).with_suffix(suffix)  # add suffix

    path.parent.mkdir(exist_ok=True, parents=True)  # create parent dir

    if os.path.exists(path):
        _rename_old_file(path)
    return path
