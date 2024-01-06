import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load your BERT model
model_bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)
model_bert.load_state_dict(torch.load("path/to/your/saved/model.pth"))
model_bert.eval()

# Load your label mapping
label_mapping = {
    0: "gender_bias",
    1: "religion_bias",
    2: "country_bias",
    3: "non_bias",
}

# Streamlit app
st.title("Bias Prediction App")

# Input text area for user to enter text
text_to_classify_bert = st.text_area("Enter text for bias prediction", "")

# Tokenize input text and make prediction
if st.button("Predict Bias"):
    if text_to_classify_bert:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        input_ids_bert = tokenizer(text_to_classify_bert, padding='max_length', truncation=True, max_length=128, return_tensors='pt')['input_ids']
        attention_mask_bert = tokenizer(text_to_classify_bert, padding='max_length', truncation=True, max_length=128, return_tensors='pt')['attention_mask']

        with torch.no_grad():
            output_bert = model_bert(input_ids_bert, attention_mask=attention_mask_bert)

        predicted_label_bert = torch.argmax(output_bert.logits, dim=1).item()
        predicted_bias_bert = label_mapping[predicted_label_bert]

        st.success(f"Predicted Bias: {predicted_bias_bert}")
    else:
        st.warning("Please enter some text for prediction.")
