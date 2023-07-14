from src.tar.align import Aligner


def test_alinger():
    aliner = Aligner()
    sentence_en = "Madam President, I would like to confine my remarks to Alzheimer's disease."
    sentence_de = "Frau Präsidentin, ich möchte meine Bemerkungen auf die Alzheimer-Krankheit beschränken."
    mapping, tokens_en, tokens_de = aliner(sentence_en, sentence_de)

    for src, trg in mapping:
        print(f"{sentence_en[src]}\t->\t{sentence_de[trg]}")
