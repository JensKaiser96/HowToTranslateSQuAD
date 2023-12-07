from difflib import SequenceMatcher
from typing import Optional


def fuzzy_match(_str: str, options: list[str]) -> Optional[str]:
    """
    returns the string out of the options which is most similar to the input string,
    returns None on empty args or non unique solution
    """
    if not _str or not options:
        return None

    similarities = [SequenceMatcher(a=_str.lower(), b=option.lower()).ratio() for option in options]
    max_value = max(similarities)
    max_index = similarities.index(max_value)

    if similarities.count(max_value) == 1:
        return options[max_index]
    return None
