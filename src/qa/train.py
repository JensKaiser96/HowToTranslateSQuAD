from transformers import TrainingArguments, Trainer

from src.io.filepaths import Models
from src.io.utils import str_to_safe_path
from src.qa.gelectra_quad import Gelectra
from src.qa.quad import QUAD
from src.utils.logging import get_logger

logger = get_logger(__name__)

# TODO, make pipeline modular: -> train dataset var, (model) var, save path var
logger.info("Loading Model ...")
gelectra_base = Gelectra.Base

logger.info("Preparing Datasets ...")
train_dataset = QUAD(QUAD.Datasets.Squad1.Translated.Raw.TRAIN_CLEAN).as_hf_dataset(gelectra_base.tokenizer.model)
validation_dataset = QUAD(QUAD.Datasets.GermanQuad.TEST).as_hf_dataset(gelectra_base.tokenizer.model)

train_dataloader = DataLoader(
    tokenized_train_dataset,
    shuffle=True,
    collate_fn=default_data_collator,
    batch_size=254,
)
eval_dataloader = DataLoader(
    tokenized_validation_dataset, collate_fn=default_data_collator, batch_size=8
)

model = AutoModelForQuestionAnswering.from_pretrained("deepset/gelectra-large")
trained_model_name = "test"
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
    tokenizer=gelectra_base.tokenizer.model
)
logger.info("Training Model ...")
trainer.train()

logger.info("Saving Model ...")
trainer.save_model(str_to_safe_path(Models.QA.Gelectra.raw_clean))
