import torch
from transformers.tokenization_utils_base import BatchEncoding
from transformers import XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer

from src.io.filepaths import Alignment
from src.tar.utils import Span, sinkhorn
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Aligner:
    def __init__(self):
        # load and set model config
        model_config = XLMRobertaConfig.from_pretrained(Alignment.config)
        model_config.output_hidden_states = True
        model_config.return_dict = False

        # load alignment model
        self.model = XLMRobertaModel.from_pretrained(
                Alignment.model_path, config=model_config)

        # load tokenizer
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(Alignment.bpq)

    # TODO: make it work with batches
    def __call__(self, sentence1: str, sentence2: str
                 ) -> list[tuple[int, int]]:
        # tokenize sentences
        encoding = self.tokenizer(sentence1, sentence2, return_tensors="pt")
        span1, span2 = self.extract_spans(encoding)

        with torch.no_grad():
            _, _, outputs = self.model(**encoding)
        best_alignment_output = outputs[8]  # layer 8 has the best alignment
        # I don't get why this is done, but its in the reference code
        sinkhorn_input = torch.bmm(
                best_alignment_output,
                best_alignment_output.transpose(1, 2))
        # The sinkhorn algorithm returns the alignment pairs
        return sinkhorn(None, sinkhorn_input, span1, span2)

    def extract_spans(self, encoding: BatchEncoding) -> tuple[Span, Span]:
        """
        extracts spans from an encoding of two sentences, the span start index
        is inclusive and the span end is exclusive, e.g.:
            Span(2,5) includes the elements 2, 3, and 4 (not 5)
        It is expected that the encoding has the following format in its
        input_ids tensor:
            [[BOS, <sentence1>, EOS, EOS, <sentence2>, EOS]]
        """
        BOS = self.tokenizer.bos_token_id
        EOS = self.tokenizer.eos_token_id

        ids = list(encoding.input_ids.flatten())

        # check if sequence is as expected
        if not ids[0] == BOS:
            raise ValueError(
                f"Expected sequence to start with [BOS] token (id:{BOS}), "
                f"but sequence starts with id:{ids[0]}.")
        if not ids[-1] == EOS:
            raise ValueError(
                f"Expected sequence to end with [EOS] token (id:{EOS}), "
                f"but sequence ends with id:{ids[0]}.")
        if not list(ids).count(EOS) == 3:
            raise ValueError(
                f"Expected sequence to have exactly three occurences of the "
                f"[EOS] token (id:{EOS}), but counted {list(ids).count(EOS)} "
                f"instead.")

        first_EOS = ids.index(EOS)

        if not ids[first_EOS + 1] == EOS:
            raise ValueError(
                f"Expected sequence to have the second [EOS] directly follow "
                f" the first [EOS] (id:{EOS}), but the token after the first "
                f"[EOS] has id: {ids[first_EOS + 1]} instead.")

        source_span = Span(1, first_EOS)
        target_span = Span(first_EOS + 2, -1)

        return source_span, target_span
