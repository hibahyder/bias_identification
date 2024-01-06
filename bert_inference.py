import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score

# Test a sample text
text_to_classify_bert = "It could all be set up. I don't know if I believe Auschwitz ever really happened."
input_ids_bert = tokenizer(text_to_classify_bert, padding='max_length', truncation=True, max_length=128, return_tensors='pt')['input_ids']
attention_mask_bert = tokenizer(text_to_classify_bert, padding='max_length', truncation=True, max_length=128, return_tensors='pt')['attention_mask']

with torch.no_grad():
    model_bert.eval()
    output_bert = model_bert(input_ids_bert, attention_mask=attention_mask_bert)

# Determine the predicted label for BERT model
predicted_label_bert = torch.argmax(output_bert.logits, dim=1).item()

# Map the predicted label back to the label name using the label_mapping
for label, value in label_mapping.items():
    if value == predicted_label_bert:
        predicted_bias_bert = label
        break

# Print the prediction for BERT model
print(f"{text_to_classify_bert}: {predicted_bias_bert}")
