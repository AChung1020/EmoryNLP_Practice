from transformers import BertForSequenceClassification, BertTokenizer, pipeline
import torch
import pandas as pd
from datasets import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
import optuna
import numpy as np

print(torch.version.cuda)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

def encode(examples):
    return tokenizer(examples['sentence'], truncation=True, padding='max_length')

#data processing/tokenizing
df_train = pd.read_csv('data/cola_train.csv', sep=',')
df_train= df_train.drop('human', axis=1) #human annotatino column   
df_train = df_train.drop('source', axis=1) #source of sentence column

df_dev = pd.read_csv('data/cola_dev.csv', sep=',')
df_dev = df_dev.drop('human', axis=1)
df_dev = df_dev.drop('source', axis=1)

train_dataset = Dataset.from_pandas(df_train).map(encode, batched=True) #make into Dataset Object
val_dataset = Dataset.from_pandas(df_dev).map(encode, batched=True)

train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label']) #correct format for BERT
val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

from datasets import load_metric

accuracy_metric = load_metric('accuracy')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer) #collator

training_args = TrainingArguments(
    output_dir="results",
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=50,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    adam_beta1=0.9,
    adam_beta2=0.999,
    optim="adamw_hf",
    learning_rate=2e-05,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

model.save_pretrained("./BERT_trained")
tokenizer.save_pretrained("./BERT_trained")