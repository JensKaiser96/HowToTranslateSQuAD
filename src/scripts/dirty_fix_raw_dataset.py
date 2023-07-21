"""
dirty fix of raw quad
paragraphs are lists directly containing the paragraph data instead of dict with
title, paragraph keys
"""
from src.qa.quad import QUAD
from src.tar.translate import Translator

raw_broken = QUAD(QUAD.Datasets.Squad1.Translated.Raw.TRAIN)
raw_fixed = QUAD()
squad1 = QUAD(QUAD.Datasets.Squad1.TRAIN)

raw_iterator = iter(raw_broken.data._data)
squad1_iterator = iter(squad1._data["data"])

t = Translator()

for paragraph in squad1_iterator:
    translated_title = t.en2de(paragraph["title"])
    translated_paragraph = next(raw_iterator)
    raw_fixed.data._data.append({"title": translated_title,
                                "paragraphs": translated_paragraph})

raw_fixed.save("raw_fixed.json")
