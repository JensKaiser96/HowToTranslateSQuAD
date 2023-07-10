import logging
import warnings
import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)


class Translator:
    _en2de_model_name = "transformer.wmt19.en-de"
    _de2en_model_name = "transformer.wmt19.de-en"
    _repo = "pytorch/fairseq"
    _tokenizer = "moses"
    _bpe = "fastbpe"  # "subword_nmt"

    def __init__(self):
        self._en2de_model = None
        self._de2en_model = None

        # disable info logging
        fairseq_loggers = (
            "fairseq.tasks.fairseq_task",
            "fairseq.tasks.translation",
            "fairseq.models.fairseq_model",
            "fairseq.file_utils")
        for fairseq_logger in fairseq_loggers:
            logging.getLogger(fairseq_logger).setLevel(logging.WARN)

    def _load_model(self, model_name: str):
        logger.info(f"Loading translation model:'{model_name}' ...")
        # TODO: properly fix model loading instead of muting warnings!!!
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = torch.hub.load(
                    self._repo, model_name,
                    checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
                    tokenizer=self._tokenizer, bpe=self._bpe)
        model.eval()  # disable dropout
        model.cuda()  # move model to GPU
        return model

    def en2de(self, text: str) -> str:
        if not self._en2de_model:
            self._en2de_model = self._load_model(self._en2de_model_name)
        return self._en2de_model.translate(text)

    def de2en(self, text: str) -> str:
        if not self._de2en_model:
            self._de2en_model = self._load_model(self._de2en_model_name)
        return self._de2en_model.translate(text)
