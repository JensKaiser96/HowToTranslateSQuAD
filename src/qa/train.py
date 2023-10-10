from transformers import TrainingArguments, Trainer

from src.io.utils import str_to_safe_path
from src.qa.dataset import Dataset
from src.qa.qamodel import QAModel
from src.utils.logging import get_logger

logger = get_logger(__name__)
# https://huggingface.co/docs/transformers/tasks/question_answering
# https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments


def train(
    base_model: QAModel,
    train_dataset: Dataset,
    save_path,
    validation_dataset: Dataset = None,
    **kwargs
):
    logger.info("Preparing Datasets ...")
    train_dataset = train_dataset.as_hf_dataset(
        tokenizer=base_model.tokenizer.model,
        max_length=base_model.model.config.max_position_embeddings,
    )
    if validation_dataset is not None:
        validation_dataset = validation_dataset.as_hf_dataset(
            tokenizer=base_model.tokenizer.model,
            max_length=base_model.model.config.max_position_embeddings,
        )

    # https://github.com/google-research/electra/blob/master/configure_finetuning.py
    """
    self.num_train_epochs = 2.0
    self.learning_rate = 1e-4
    self.weight_decay_rate = 0.01
    self.layerwise_lr_decay = 0.8  # if > 0, the learning rate for a layer is
                                   # lr * lr_decay^(depth - max_depth) i.e.,
                                   # shallower layers have lower learning rates
    self.warmup_proportion = 0.1  # how much of training to warm up the LR for
    self.train_batch_size = 32
    """
    default_args = dict(
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        remove_unused_columns=False,
        evaluation_strategy="no",
        save_strategy="no",
        learning_rate=1e-5,
        num_train_epochs=2,
        weight_decay=0.01,
        fp16=True,
    )

    # overwrite/extend default TrainingArguments
    for key, value in kwargs.items():
        default_args[key] = value

    args = TrainingArguments(output_dir=save_path + "/checkpoints", **default_args)
    trainer = Trainer(
        model=base_model.model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=base_model.tokenizer.model,
    )
    logger.info("Training Model ...")
    trainer.train()

    logger.info("Saving Model ...")
    trainer.save_model(str_to_safe_path(save_path, dir_ok=True))
