import torch
from transformers import XLMRobertaConfig, XLMRobertaModel

from src.io.filepaths import Alignment
from src.tar.utils import Span, Direction
from src.tar.tokenize import Tokenizer
from src.math.matrix import dimensionalwise_normalize
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Aligner:
    # load and set model config
    _model_config = XLMRobertaConfig.from_pretrained(Alignment.config)
    _model_config.output_hidden_states = True
    _model_config.return_dict = False

    # load alignment model
    model = XLMRobertaModel.from_pretrained(
        Alignment.model_path, config=_model_config)

    @classmethod
    def __call__(cls,
                 source_text: str,
                 target_text: str,
                 direction: Direction = Direction.forwards,
                 ) -> list[tuple[int, int]]:
        """
        returns the alignment between the tokens in text_1 and sentence2
        as well as the tokens in source_text and target_text, without [BOS] and
        [EOS], combinded with their index in the text to distiglish between
        tokens with the same string representation
        """
        # get encodings, aligner output and spans
        encoding = Tokenizer.encode(source_text, target_text)
        output = cls._get_model_output(encoding)
        source_span, target_span = Tokenizer.split_encoding(encoding)
        alignments = cls._get_alignments_from_model_output(
            output, source_span, target_span, direction)

        # add position (idx) to each token before returning the token list
        source_tokens = [(idx, token) for idx, token in enumerate(
            Tokenizer.decode(source_span(encoding)))]
        target_tokens = [(idx, token) for idx, token in enumerate(
            Tokenizer.decode(target_span(encoding)))]

        for source, target in alignments:
            logger.debug(
                f"{source_tokens[source][1]}\t->\t{target_tokens[target][1]}")

        return alignments, source_tokens, target_tokens

    @classmethod
    def _get_model_output(cls, encoding: dict = None) -> torch.Tensor:
        with torch.no_grad():
            _, _, outputs = cls.model(**encoding)
        output = outputs[8]  # layer 8 has the best alignment
        # I don't get why this is done, but its in the reference code
        return torch.bmm(output, output.transpose(1, 2))[0]

    @classmethod
    def _get_alignments_from_model_output(
            cls,
            output: torch.Tensor,
            source_span: Span,
            target_span: Span,
            direction: Direction):
        # crop array to exclude tokens outside the spans
        relevant_output = output[source_span.start: source_span.end,
                                 target_span.start: target_span.end]
        normalized_output = dimensionalwise_normalize(relevant_output)

        # actual alignment
        if direction == Aligner.Direction.forwards:
            best_match = normalized_output.argmax(axis=1)
            return [[i, t] for i, t in enumerate(best_match)]

        elif direction == Aligner.Direction.backwards:
            best_match = normalized_output.argmax(axis=0)
            return [[i, t] for i, t in enumerate(best_match)]

        elif direction == Aligner.Direction.bidirectional:
            # sinkhorn algorithm
            m, n = normalized_output.size()
            forward = torch.eye(n)[normalized_output.argmax(dim=1)]
            backward = torch.eye(m)[normalized_output.argmax(dim=0)]
            inter = forward * backward.T
            return inter.nonzero()

        else:
            raise ValueError(
                f"Direction argument must be a Direction, one of: "
                f"{Aligner.Direction._member_names_}")
