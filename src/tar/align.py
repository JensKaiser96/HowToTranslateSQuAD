import torch
from transformers.tokenization_utils_base import BatchEncoding
from transformers import XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer
from typing import Sequence, Union

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
    def __call__(self, sentence1: str, sentence2: str, return_tokens=False,
                 ) -> list[tuple[int, int]]:
        """
        returns the alignment between the tokens in sentence1 and sentence2
        as well as the tokens in sentence1 and sentence2 (without BOS and EOS)
        """
        # tokenize sentences
        encoding = self.tokenizer(sentence1, sentence2, return_tensors="pt")
        span1, span2 = self.extract_spans(encoding)

        with torch.no_grad():
            _, _, outputs = self.model(**encoding)
        best_alignment_output = outputs[8]  # layer 8 has the best alignment
        # I don't get why this is done, but its in the reference code
        sinkhorn_input = torch.bmm(
                best_alignment_output,
                best_alignment_output.transpose(1, 2))[0]
        # The sinkhorn algorithm returns the alignment pairs
        sinkhorn_output = sinkhorn(sinkhorn_input, span1, span2)

        # the output is based on the encoding representation, i.e. first the
        # [BOS] token then, sentence2, [EOS], [EOS], sentence2, [EOS]
        # but we want the mapping between sentence1 and sentence2 when they
        # both start at index 0, so the span.start is substracted
        alignments = [(source - span1.start, target - span2.start)
                      for source, target in sinkhorn_output]
        alignments = self.extrapolate_alignment(alignments)
        return (alignments,
                self.decode(span1(encoding)),
                self.decode(span2(encoding)))

    def decode(self, sequence: Sequence[Union[int, torch.Tensor]]) -> list[str]:
        return [self.tokenizer.decode(token_id) for token_id in sequence]

    def extrapolate_alignment(alignments):
        alignments = alignments.copy()

        expected_source_index = 0
        last_target_index = 0

        for source_id, target_id in alignments:
            if source_id == expected_source_index:
                last_target_index = target_id
                continue
            extrapolated_match = (expected_source_index, last_target_index)
            alignments.insert(expected_source_index, extrapolated_match)

        return alignments

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
        logger.debug(f"Extracting spans from:\n {ids}.")

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
        target_span = Span(first_EOS + 2, len(ids)-1)

        logger.debug(f"First span is:\n{source_span(ids)}\n"
                     f"Secod span is:\n{target_span(ids)}")

        return source_span, target_span
