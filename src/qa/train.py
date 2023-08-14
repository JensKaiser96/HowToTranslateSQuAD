import datasets
from torch.utils.data import DataLoader
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import default_data_collator

from src.io.filepaths import Datasets
from src.io.utils import str_to_safe_path
from src.qa.gelectra_quad import Gelectra
from src.qa.train_util import prepare_train_features, flatten_quad
from src.utils.logging import get_logger

logger = get_logger(__name__)

tokenizer = Gelectra.tokenizer.model

train_dataset = datasets.load_dataset("json", data_files=Datasets.Squad1.Translated.Raw.TRAIN_CLEAN, field="data", split="train")
train_dataset = train_dataset.map(flatten_quad, batched=True, remove_columns=train_dataset.column_names)
tokenized_train_dataset = train_dataset.map(prepare_train_features, batched=True, remove_columns=train_dataset.column_names)
tokenized_train_dataset.set_format("torch")

validation_dataset = datasets.load_dataset("json", data_files=Datasets.GermanQuad.TEST, field="data", split="train")
validation_dataset = validation_dataset.map(flatten_quad, batched=True, remove_columns=validation_dataset.column_names)
tokenized_validation_dataset = validation_dataset.map(prepare_train_features, batched=True, remove_columns=validation_dataset.column_names)
tokenized_validation_dataset.set_format("torch")

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
    model=model,
    args=args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_validation_dataset,
    tokenizer=tokenizer
)
trainer.train()

trainer.save_model(str_to_safe_path("data/models/gelectra/raw_clean/"))

