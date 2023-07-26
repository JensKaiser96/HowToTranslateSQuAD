import torch
from transformers import XLMRobertaConfig, XLMRobertaModel

from src.io.filepaths import Alignment
from src.tar.utils import Direction
from src.nlp_tools.span import Span
from src.tar.tokenize import Tokenizer
from src.math.matrix import dimensionalwise_normalize
from src.utils.logging import get_logger

logger = get_logger(__name__)

# load and set model config
_model_config = XLMRobertaConfig.from_pretrained(Alignment.config)
_model_config.output_hidden_states = True
_model_config.return_dict = False

# load alignment model
model = XLMRobertaModel.from_pretrained(
    Alignment.model_path, config=_model_config)


def align(source_text: str, target_text: str,
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
    output = _get_model_output(encoding)
    source_span, target_span = Tokenizer.split_encoding(encoding)
    alignments = _get_alignments_from_model_output(
        output, source_span, target_span, direction)

    # add position (idx) to each token before returning the token list
    source_tokens = Tokenizer.decode(source_span(encoding))
    target_tokens = Tokenizer.decode(target_span(encoding))

    for source, target in alignments:
        logger.debug(f"{source_tokens[source]}\t->\t{target_tokens[target]}")

    return alignments, source_tokens, target_tokens


def _get_model_output(encoding: dict = None) -> torch.Tensor:
    with torch.no_grad():
        _, _, outputs = model(**encoding)
    output = outputs[8]  # layer 8 has the best alignment
    # I don't get why this is done, but its in the reference code
    return torch.bmm(output, output.transpose(1, 2))[0]


def _get_alignments_from_model_output(
        output: torch.Tensor,
        source_span: Span,
        target_span: Span,
        direction: Direction):
    # crop array to exclude tokens outside the spans
    relevant_output = output[source_span.start: source_span.end,
                             target_span.start: target_span.end]
    normalized_output = dimensionalwise_normalize(relevant_output)

    # actual alignment
    if direction == Direction.forwards:
        best_match = normalized_output.argmax(axis=1)
        return [[i, t] for i, t in enumerate(best_match)]

    elif direction == Direction.backwards:
        best_match = normalized_output.argmax(axis=0)
        return [[i, t] for i, t in enumerate(best_match)]

    elif direction == Direction.bidirectional:
        # sinkhorn algorithm
        m, n = normalized_output.size()
        forward = torch.eye(n)[normalized_output.argmax(dim=1)]
        backward = torch.eye(m)[normalized_output.argmax(dim=0)]
        inter = forward * backward.T
        return inter.nonzero()

    else:
        raise ValueError(
            f"Direction argument must be a Direction, one of: "
            f"{Direction._member_names_}")
