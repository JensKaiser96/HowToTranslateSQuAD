from transformers import TrainingArguments, Trainer

from src.io.utils import str_to_safe_path
from src.qa.dataset import Dataset
from src.qa.gelectra import Gelectra
from src.utils.logging import get_logger

logger = get_logger(__name__)
# https://huggingface.co/docs/transformers/tasks/question_answering


def train(
    base_model: Gelectra,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    save_path,
    name="",
    **kwargs
):
    logger.info("Preparing Datasets ...")
    train_dataset = train_dataset.as_hf_dataset(
        tokenizer=base_model.tokenizer.model,
        max_length=base_model.model.config.max_position_embeddings,
    )
    validation_dataset = validation_dataset.as_hf_dataset(
        tokenizer=base_model.tokenizer.model,
        max_length=base_model.model.config.max_position_embeddings,
    )

    default_args = dict(
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        remove_unused_columns=False,
        evaluation_strategy="no",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=True,
    )

    # overwrite/extend default TrainingArguments
    for key, value in kwargs.items():
        default_args[key] = value

    # TODO: which hyper-parameters do i use??!??!?
    args = TrainingArguments( name, **default_args)
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
    trainer.save_model(str_to_safe_path(save_path))
