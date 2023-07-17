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
    def __call__(self, source_text: str, target_text: str
                 ) -> list[tuple[int, int]]:
        """
        returns the alignment between the tokens in text_1 and sentence2
        as well as the tokens in sentence1 and sentence2 (without BOS and EOS)
        """
        # tokenize sentences
        encoding = self.tokenizer(source_text, target_text,
                                  return_tensors="pt")
        source_span, target_span = self.extract_spans(encoding)

        with torch.no_grad():
            _, _, outputs = self.model(**encoding)
        best_alignment_output = outputs[8]  # layer 8 has the best alignment
        # I don't get why this is done, but its in the reference code
        sinkhorn_input = torch.bmm(
                best_alignment_output,
                best_alignment_output.transpose(1, 2))[0]
        # The sinkhorn algorithm returns the alignment pairs
        sinkhorn_output = sinkhorn(sinkhorn_input, source_span, target_span)

        # the output is based on the encoding representation, i.e. first the
        # [BOS] token then, sentence2, [EOS], [EOS], sentence2, [EOS]
        # but we want the mapping between sentence1 and sentence2 when they
        # both start at index 0, so the span.start is substracted
        alignments = [(source - source_span.start, target - target_span.start)
                      for source, target in sinkhorn_output]
        return (alignments,
                self.decode(source_span(encoding)),
                self.decode(target_span(encoding)))

    def retrive(self, source_text: str, source_span: Span,
                target_text: str) -> Span:
        """
        Given a source text, its answer span and the translation of the
        source text (target_text), this method returns the answer span
        of the target_text
        """
        # get mapping between source_text and target_text
        mapping, source_tokens, target_tokens = self()

        # find tokens corresponding to the span.
        source_span_tokens = []

        # mapping [(0, 0), (1, 2), ...]

    @staticmethod
    def surface_token_mapping(text: str, tokens: list):
        mapping = {}
        curser_pos = 0
        for index, token in enumerate(tokens):
            span = Span(curser_pos, curser_pos + len(token))
            if token != span(text):
                raise ValueError(
                        f"Expected token '{token}' to be at {span.start}: "
                        f"{span.end} in \n'{text}'\n, was: \n'{span(text)}'")
            curser_pos += span.end
            if text[curser_pos] == " ":
                curser_pos += 1
            mapping[(index, token)] = span
        return mapping


    def decode(self, sequence: Sequence[Union[int, torch.Tensor]]) -> list[str]:
        return [self.tokenizer.decode(token_id) for token_id in sequence]

    def extract_spans(self, encoding: BatchEncoding) -> tuple[Span, Span]:
        """
        extracts spans from an encoding of two texts, the span start index
        is inclusive and the span end is exclusive, e.g.:
            Span(2,5) includes the elements 2, 3, and 4 (not 5)
        It is expected that the encoding has the following format in its
        input_ids tensor:
            [[BOS, <source_text>, EOS, EOS, <target_text>, EOS]]
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
                f"but sequence ends with id:{ids[-1]}.")
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

        return source_span, target_span
