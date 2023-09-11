from transformers import TrainingArguments, Trainer

from src.io.filepaths import Models
from src.io.utils import str_to_safe_path
from src.qa.dataset import Dataset
from src.qa.gelectra import Gelectra
from src.utils.logging import get_logger

logger = get_logger(__name__)

# TODO, make pipeline modular: -> train dataset var, (model) var, save path var
logger.info("Loading Model ...")
gelectra_base = Gelectra.Base

logger.info("Preparing Datasets ...")
train_dataset = Dataset.Raw.TRAIN_CLEAN.as_hf_dataset(
    tokenizer=gelectra_base.tokenizer.model,
    max_length=gelectra_base.model.config["max_position_embeddings"],
)
validation_dataset = Dataset.GermanQUAD.TEST.as_hf_dataset(
    gelectra_base.tokenizer.model
)

trained_model_name = "raw_clean"
# TODO, get train args from GermanQuad Guys
args = TrainingArguments(
    trained_model_name,
    remove_unused_columns=False,
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
)
trainer = Trainer(
    model=gelectra_base.model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=gelectra_base.tokenizer.model,
)
logger.info("Training Model ...")
trainer.train()

logger.info("Saving Model ...")
trainer.save_model(str_to_safe_path(Models.QA.Gelectra.raw_clean))
