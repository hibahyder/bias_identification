import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score

dataset_path = "bias_project_dataset.csv"
df = pd.read_csv(dataset_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Extract text and labels from your dataset
texts = df['TEXT'].tolist()
text_labels = df['LABEL'].tolist()

# Create a label mapping from text labels to integers
label_mapping = {
    "gender_bias": 0,
    "religion_bias": 1,
    "country_bias": 2,
    "non_bias": 3,
}

# Encode text labels as integers using the label mapping
labels = [label_mapping[label] for label in text_labels]

# Tokenize the text data and create input tensors
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
input_ids = []
attention_masks = []

for text in texts:
    encoding = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    input_ids.append(encoding['input_ids'])
    attention_masks.append(encoding['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# Split the dataset into training and validation sets
train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels = train_test_split(
    input_ids, attention_masks, labels, test_size=0.2, random_state=42
)

# BERT Model
num_classes = 4
model_bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes)
model_bert.to(device)
# Create data loaders for BERT model
batch_size = 8
train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(val_inputs, val_masks, val_labels)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Training settings for BERT model
criterion_bert = nn.CrossEntropyLoss()
optimizer_bert = AdamW(model_bert.parameters(), lr=2e-5)
num_epochs_bert = 3

# Training loop for BERT model
for epoch in range(num_epochs_bert):
    model_bert.train()
    total_loss_bert = 0.0
    for inputs, masks, labels in train_dataloader:
        optimizer_bert.zero_grad()
        # move data to device (gpu/cpu)
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
        outputs_bert = model_bert(inputs, attention_mask=masks, labels=labels)
        # training loss
        loss_bert = outputs_bert.loss
        # backpropogation
        loss_bert.backward()
        optimizer_bert.step()
        total_loss_bert += loss_bert.item()

    average_loss_bert = total_loss_bert / len(train_dataloader)

    # Validation for BERT model
    model_bert.eval()
    val_losses_bert = []
    val_predictions_bert = []

    with torch.no_grad():
        for inputs, masks, labels in val_dataloader:
            # move data to device (gpu / cpu)
            inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
            # make prediction
            outputs_bert = model_bert(inputs, attention_mask=masks, labels=labels)
            # validation loss
            val_losses_bert.append(outputs_bert.loss)
            # validation accuracy
            val_predictions_bert.extend(outputs_bert.logits.argmax(dim=1).tolist())

    val_loss_bert = torch.mean(torch.tensor(val_losses_bert))

    # Convert predictions to numpy array
    val_predictions_bert = torch.tensor(val_predictions_bert)
    val_labels_tensor = val_labels.clone().detach()

    # Calculate accuracy for BERT model
    accuracy_bert = accuracy_score(val_labels_tensor.numpy(), val_predictions_bert.numpy())
    print(f"Epoch {epoch + 1}/{num_epochs_bert} - BERT Model - Training Loss: {average_loss_bert:.4f} - Validation Loss: {val_loss_bert:.4f} - Val Accuracy: {accuracy_bert:.4f}")

    torch.save(model_bert, "bert_model.pth")
