import torch
import logging

from src.tools.logging import get_logger

logger = get_logger(__name__)


class Translator:
    en2de_model_name = "transformer.wmt19.en-de"
    de2en_model_name = "transformer.wmt19.de-en"
    _repo = "pytorch/fairseq"
    _tokenizer = "moses"
    _bpe = "fastbpe"  # "subword_nmt"

    def __init__(self):
        self.en2de_model = None
        self.de2en_model = None

        # disable info logging
        logging.getLogger("fairseq.tasks.fairseq_task").setLevel(logging.WARN)
        logging.getLogger("fairseq.tasks.translation").setLevel(logging.WARN)
        logging.getLogger("fairseq.mdoels.fairseq_model").setLevel(logging.WARN)

    def _load_model(self, model_name: str):
        logger.info(f"Loading translation model:'{model_name}' ...")
        model = torch.hub.load(
                self._repo, model_name,
                checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
                tokenizer=self._tokenizer, bpe=self._bpe)
        model.eval()  # disable dropout
        model.cuda()  # move model to GPU
        return model

    def en2de(self, text: str) -> str:
        if not self.en2de_model:
            self.en2de_model = self._load_model(self.en2de_model_name)
        return self.en2de_model.translate(text)

    def de2en(self, text: str) -> str:
        if not self.de2en_model:
            self.de2en_model = self._load_model(self.de2en_model_name)
        return self.de2en_model.translate(text)
