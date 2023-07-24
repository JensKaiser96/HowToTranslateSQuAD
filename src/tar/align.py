import torch
import string
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
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(
            Alignment.model_path)

    # TODO: make it work with batches
    def bidirectional_alignment(self, source_text: str, target_text: str
                                ) -> list[tuple[int, int]]:
        """
        returns the alignment between the tokens in text_1 and sentence2
        as well as the tokens in source_text and target_text, without [BOS] and
        [EOS], combinded with their index in the text to distiglish between
        tokens with the same string representation
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

        # add position (idx) to each token before returning the token list
        source_tokens = [(idx, token) for idx, token in enumerate(
            self.decode(source_span(encoding)))]
        target_tokens = [(idx, token) for idx, token in enumerate(
            self.decode(target_span(encoding)))]
        return alignments, source_tokens, target_tokens

    def retrive(self, source_text: str, source_span: Span,
                target_text: str) -> Span:
        """
        Given a source text, its answer span and the translation of the
        source text (target_text), this method returns the answer span
        of the target_text
        """
        # get mapping between source_text and target_text
        mapping, source_tokens_ids, target_tokens_ids = self.bidirectional_alignment(
            source_text, target_text)
        logger.debug(f"{mapping=}\n{source_tokens_ids=}\n{target_tokens_ids=}")

        # get surface token mapping for both source and target
        source_surface_token_mapping = self.surface_token_mapping(
            source_text, source_tokens_ids)
        logger.debug(f"{source_surface_token_mapping=}")
        target_surface_token_mapping = self.surface_token_mapping(
            target_text, target_tokens_ids)
        logger.debug(f"{target_surface_token_mapping=}")

        source_span_token_ids = source_surface_token_mapping.get_indices(
            source_span)
        logger.debug(f"{source_span_token_ids}")

        # get the tokens in the target span
        mapping_dict = {entry[0]: entry[1] for entry in mapping}
        logger.debug(f"{mapping_dict=}")
        target_span_tokens = [
            mapping_dict[token_id] for token_id in source_span_token_ids
            if token_id in mapping_dict]
        logger.debug(f"{target_span_tokens=}")

        target_surface_spans = target_surface_token_mapping.get_spans(
            target_span_tokens)
        logger.debug(f"{target_surface_spans=}")
        return Span.combine(target_surface_spans)

    class Mapping:
        def __init__(self):
            self._indices = []  # maybe this is not necessary
            self._span_starts = []
            self._span_ends = []

        def __len__(self):
            return len(self._indices)

        def __getitem__(self, index):
            return (self._indices[index],
                    self._span_starts[index],
                    self._span_ends[index])

        def get_indices(self, span: Span):
            return [index for index in self._indices
                    if self._span_starts[index] >= span.start
                    and self._span_ends[index] <= span.end]

        def get_spans(self, indices: list[int]):
            return [self.get_span(index) for index in indices]

        def append(self, index: int, span: Span):
            self._indices.append(index)
            self._span_starts.append(span.start)
            self._span_ends.append(span.end)

    @classmethod
    def surface_token_mapping(cls, text: str, tokens: list) -> Mapping:
        """
        creates a bi-directional mapping between the surface level spans of
        words in ´text´ and (idx, token)
        """
        mapping = cls.Mapping()
        curser_pos = 0
        for index, token in tokens:
            # advance curser if the next char is a whitespace.
            while text[curser_pos: curser_pos + 1] in string.whitespace:
                curser_pos += 1
            # create span over current token
            span = Span(curser_pos, curser_pos + len(token))
            # check if content of the span matches with the token
            if token == span(text):
                mapping.append(index, span)
            else:
                raise ValueError(
                    f"Expected token '{token}' to be at {span.start}: "
                    f"{span.end} in \n'{text}'\n, was: \n'{span(text)}'")
            # move curser to the end of the span
            curser_pos = span.end
        return mapping

    def decode(self,
               sequence: Sequence[Union[int, torch.Tensor]]) -> list[str]:
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
