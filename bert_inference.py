import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model_bert = torch.load("bert_model.pth",map_location=torch.device('cpu'))
# Test a sample text
text_to_classify_bert = "I think India is really bad"
input_ids_bert = tokenizer(text_to_classify_bert, padding='max_length', truncation=True, max_length=128, return_tensors='pt')['input_ids']
attention_mask_bert = tokenizer(text_to_classify_bert, padding='max_length', truncation=True, max_length=128, return_tensors='pt')['attention_mask']

with torch.no_grad():
    model_bert.eval()
    output_bert = model_bert(input_ids_bert, attention_mask=attention_mask_bert)
# Determine the predicted label for BERT model
predicted_label_bert = torch.argmax(output_bert.logits, dim=1).item()

dataset_path = "bias_project_dataset.csv"
df = pd.read_csv(dataset_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


texts = df['TEXT'].tolist()
text_labels = df['LABEL'].tolist()

# Map the predicted label back to the label name using the label_mapping
label_mapping = {
    "gender_bias": 0,
    "religion_bias": 1,
    "country_bias": 2,
    "non_bias": 3,
}

# Encode text labels as integers using the label mapping
labels = [label_mapping[label] for label in text_labels]
for label, value in label_mapping.items():
    if value == predicted_label_bert:
        predicted_bias_bert = label
        break

# Print the prediction for BERT model
print(f"{text_to_classify_bert}: {predicted_bias_bert}")
