from src.nlp_tools.span import Span
from src.utils.logging import get_logger
from src.tar.align import align
from src.tar.tokenize import Tokenizer

logger = get_logger(__name__)


def retrive(source_text: str, source_span: Span,
            target_text: str) -> Span:
    """
    Given a source text, its answer span and the translation of the
    source text (target_text), this method returns the answer span
    of the target_text
    """
    # get mapping between source_text and target_text
    alignment, source_tokens, target_tokens = align(
        source_text, target_text)
    logger.debug(f"{alignment=}\n{source_tokens=}\n{target_tokens=}")

    # get surface token mapping for both source and target
    source_surface_token_mapping = Tokenizer.surface_token_mapping(
        source_text, source_tokens)
    target_surface_token_mapping = Tokenizer.surface_token_mapping(
        target_text, target_tokens)

    source_span_token_ids = [index for index, span
                             in enumerate(source_surface_token_mapping)
                             if span.is_subspan(source_span)]
    logger.debug(f"{source_span_token_ids}")

    # get the tokens in the target span
    # todo: this should work more elegant
    mapping_dict = {entry[0]: entry[1] for entry in alignment}
    # logger.debug(f"{mapping_dict=}")
    target_span_tokens = [mapping_dict[token_id] for token_id
                          in source_span_token_ids
                          if token_id in mapping_dict]
    logger.debug(f"{target_span_tokens=}")

    target_surface_spans = [span for index, span
                            in enumerate(target_surface_token_mapping)
                            if index in target_span_tokens]
    logger.debug(f"{target_surface_spans=}")
    return Span.combine(target_surface_spans)
