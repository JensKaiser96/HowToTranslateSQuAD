import torch
from transformers import XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from src.io.filepaths import Models
from src.math.matrix import dimensionalwise_normalize
from src.nlp_tools.span import Span
from src.nlp_tools.token import Tokenizer
from src.tar.utils import Direction
from src.utils.logging import get_logger

logger = get_logger(__name__)

# load and set model config
_model_config = XLMRobertaConfig.from_pretrained(Models.Alignment.config)
_model_config.output_hidden_states = True
_model_config.return_dict = False

# load alignment model
model = XLMRobertaModel.from_pretrained(
    Models.Alignment.model_path, config=_model_config)

tokenizer = Tokenizer(
    XLMRobertaTokenizer.from_pretrained(Models.Alignment.model_path))


def align(source_text: str, target_text: str,
          direction: Direction = Direction.forwards,
          ):
    """
    returns the alignment between the tokens in text_1 and sentence2
    as well as the tokens in source_text and target_text, without [BOS] and
    [EOS], combinded with their index in the text to distiglish between
    tokens with the same string representation
    """
    # get encodings, aligner output and spans
    encoding = tokenizer.encode(source_text, target_text)
    output = _get_model_output(encoding)
    source_span, target_span = split_encoding(encoding)
    alignments = _get_alignments_from_model_output(
        output, source_span, target_span, direction)

    # add position (idx) to each token before returning the token list
    source_tokens = tokenizer.decode(source_span(encoding))
    target_tokens = tokenizer.decode(target_span(encoding))

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


def split_encoding(encoding: BatchEncoding) -> tuple[Span, Span]:
    """
    extracts spans from an encoding of two texts, the span start index
    is inclusive and the span end is exclusive, e.g.:
        Span(2,5) includes the elements 2, 3, and 4 (not 5)
    It is expected that the encoding has the following format in its
    input_ids tensor:
        [[BOS, <source_text>, EOS, EOS, <target_text>, EOS]]
    """
    BOS = tokenizer.model.bos_token_id
    EOS = tokenizer.model.eos_token_id

    ids = list(encoding.input_ids.flatten())

    # check if sequence is as expected
    if ids[0] != BOS:
        raise ValueError(
            f"Expected sequence to start with [BOS] token (id:{BOS}), "
            f"but sequence starts with id:{ids[0]}.")
    if ids[-1] != EOS:
        raise ValueError(
            f"Expected sequence to end with [EOS] token (id:{EOS}), "
            f"but sequence ends with id:{ids[-1]}.")
    if list(ids).count(EOS) != 3:
        raise ValueError(
            f"Expected sequence to have exactly three occurences of the "
            f"[EOS] token (id:{EOS}), but counted {list(ids).count(EOS)} "
            f"instead.")

    first_eos = ids.index(EOS)

    if ids[first_eos + 1] != EOS:
        raise ValueError(
            f"Expected sequence to have the second [EOS] directly follow "
            f" the first [EOS] (id:{EOS}), but the token after the first "
            f"[EOS] has id: {ids[first_eos + 1]} instead.")

    source = Span(1, first_eos)
    target = Span(first_eos + 2, len(ids)-1)

    # Verify spans are not empty
    if source.is_empty:
        raise ValueError(
            f"Source span is not allowed to be empty. {source}")
    if target.is_empty:
        raise ValueError(
            f"Target span is not allowed to be empty. {target}")

    return source, target
