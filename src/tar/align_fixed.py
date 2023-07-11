import torch
from transformers.tokenization_utils_base import BatchEncoding
from transformers import XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer

from src.io.filepaths import Alignment
from src.tar.align import sinkhorn


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
    def __call__(self, sentence1, sentence2):
        # tokenize sentences
        inputs = self.tokenizer(sentence1, sentence2, ...)

        # <BOS> [tokens sentence1] <EOS> <EOS> [tokens sentence2] <EOS>
        # len(sentence1), len(sentence2) ?!?!?!?
        span1, span2 = extract_spans(inputs, sentence1, sentence2)

        with torch.no_grad():
            _, _, outputs = self.model(**inputs)
        best_alignment_output = outputs[8]  # layer 8 has the best alignment
        sinkhorn_input = torch.bmm(
                best_alignment_output,
                best_alignment_output.transpose(1, 2))

        return sinkhorn(None, sinkhorn_input, span1, span2)

        pass

    def combine_encodings(self, encoding1: BatchEncoding, encoding2: BatchEncoding) -> BatchEncoding:
        """
        concatinates both input_ids and attention_mask tensors
        """
        input_ids1 = encoding1.input_ids
        input_ids2 = encoding2.input_ids
        combined_input_ids = torch.cat((input_ids1, input_ids2), dim=1)

        attention_mask1 = encoding1.attention_mask
        attention_mask2 = encoding2.attention_mask2
        combined_attention_mask = torch.cat((attention_mask1, attention_mask2), dim=1)
        return BatchEncoding(
                {'input_ids': combined_input_ids,
                 'attention_mask': combined_attention_mask})

    def tokenize(self, sentence) -> BatchEncoding:
        """
        Tokenizes the input sentence, the output is a BatchEncoding Object
        which is a fancy dict with the keywords, 'input_ids' and
        'attention_mask', the return_tensors='pt' arguement assures they are
        returned as a torch.tensor Object. 'add_special_tokens=False' 
        """
        return self.tokenizer(
                sentence, return_tensors="pt", add_special_tokens=False
                )
        pass

    def align(model, tokenizer, sentence1, sentence2):
        tokenizer = XLMRobertaTokenizer.from_pretrained(Alignment.bpq)

        sentence1_encoding = tokenizer(sentence1, return_tensors="pt")
        sentence2_encoding = tokenizer(sentence2, return_tensors="pt")
        combined_encoding = combine_encoding(sentence1_encoding, sentence2_encoding)



