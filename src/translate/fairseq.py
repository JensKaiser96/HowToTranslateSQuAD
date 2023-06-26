import torch


class Translator:
    _repo = "pytorch/fairseq"
    _tokenizer = "moses"
    _bpe = "subword_nmt"

    def __init__(self):
        self.en2de_model = None
        self.de2en_model = None

    def _load_model(self, model_name: str):
        model = torch.hub.load(self._repo, model_name,
                               tokenizer=self._tokenizer, bpe=self._bpe)
        model.eval()  # disable dropout
        model.cuda()  # move model to GPU
        return model

    def _load_en2de_model(self):
        model_name = 'transformer.wmt16.en-de',
        self.en2de_model = self._load_model(model_name)

    def _load_de2_en_model(self):
        model_name = 'transformer.wmt16.de-en',
        self.en2de_model = self._load_model(model_name)

    def en2de(self, text: str) -> str:
        if not self.en2de_model:
            self._load_en2de_model()
        return self.en2de_model.translate(text)

    def de2en(self, text: str) -> str:
        if not self.de2en_model:
            self._load_de2_en_model()
        return self.de2en_model.translate(text)
