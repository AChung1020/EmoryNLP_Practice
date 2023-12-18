from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset
import numpy as np
import pandas as pd
from datasets import Dataset

model_path = "./BERT_trained"  # Path where you saved the model
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

def encode(examples):
    return tokenizer(examples['sentence'], truncation=True, padding='max_length')

df_test = pd.read_csv('data/cola_test.csv', sep=',')
val_dataset = Dataset.from_pandas(df_test).map(encode, batched=True)
val_dataset.set_format('torch', columns=['input_ids', 'attention_mask'])

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from transformers import Trainer

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    tokenizer=tokenizer
)

results = trainer.evaluate(eval_dataset=val_dataset)

predictions_output = trainer.predict(val_dataset)
predictions = predictions_output.predictions
label_ids = predictions_output.label_ids

from scipy.special import softmax

# Apply softmax to logits and get the predicted labels
pred_labels = np.argmax(softmax(predictions, axis=1), axis=1)

# Assuming df_test has an 'Id' column. If not, you'll need to add it manually
df_test['Label'] = pred_labels  # Add the predictions as a new column

# Create a new DataFrame for submission
submission_df = df_test[['Id', 'Label']]  # Select only the required columns

# Save the DataFrame as a CSV file
submission_filename = 'submission.csv'
submission_df.to_csv(submission_filename, index=False)

print(f"Submission file created: {submission_filename}")
