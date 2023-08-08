import datasets
from torch.utils.data import DataLoader
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import default_data_collator

from src.io.filepaths import Datasets
from src.qa.gelectra_quad import Gelectra


def flatten_quad(batch):
    result = {
        "id": [],
        "title": [],
        "context": [],
        "question": [],
        "answers": [],
    }
    for title, entry in zip(batch["title"], batch["paragraphs"]):
        for sub_entry, in entry:
            for qa in sub_entry["qas"]:
                result["id"].append(qa["id"])
                result["title"].append(title)
                result["context"].append(qa["context"])
                result["question"].append(qa["question"])
                result["answers"].append(qa["answers"])
        return result


train_dataset = datasets.load_dataset( "json", data_files=Datasets.Squad1.Translated.Raw.TRAIN, field="data", split="train")
train_dataset = train_dataset.map(flatten_quad, batched=True, remove_columns=train_dataset.column_names)
validation_dataset = datasets.load_dataset("json", data_files=Datasets.GermanQuad.TEST, field="data")
validation_dataset = validation_dataset.map(flatten_quad, batched=True, remove_columns=validation_dataset.column_names)
train_dataset.set_format("torch")
validation_dataset.set_format("torch")

train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=default_data_collator,
    batch_size=8,
)
eval_dataloader = DataLoader(
    validation_dataset, collate_fn=default_data_collator, batch_size=8
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

