from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from torch.utils.data import DataLoader
from transformers import default_data_collator
import datasets
from src.io.filepaths import Datasets

train_dataset = datasets.load_dataset("json", data_files=Datasets.Squad1.Translated.Raw.TRAIN, field="data")
validation_dataset = datasets.load_dataset("json", data_files=Datasets.GermanQuad.TEST, field="data")

from src.qa.gelectra_quad import Gelectra

train_dataset.set_format("torch")
validation_set = validation_dataset.remove_columns(["example_id", "offset_mapping"])
validation_set.set_format("torch")

train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=default_data_collator,
    batch_size=8,
)
eval_dataloader = DataLoader(
    validation_set, collate_fn=default_data_collator, batch_size=8
)

model = AutoModelForQuestionAnswering.from_pretrained("deepset/gelectra-large")
trained_model_name = "test"
args = TrainingArguments(
    trained_model_name,
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
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=Gelectra.model,
)
trainer.train()

