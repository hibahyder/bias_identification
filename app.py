import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load your BERT model

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model_bert = torch.load("bert_model.pth",map_location=torch.device('cpu'))
model_bert.eval()

# Load your label mapping
label_mapping = {
    0: "gender_bias",
    1: "religion_bias",
    2: "country_bias",
    3: "non_bias",
}

# Streamlit app
st.header(":green[Bias Prediction App]",divider='rainbow')

Type = st.radio(
    "Select input type",
    ["Enter a statement", "Select predefined statement"])
if Type == 'Enter a statement':
    text_to_classify_bert = st.text_area("Enter text for bias prediction", "")
else:
    text_to_classify_bert = st.selectbox(
    'Select a statement from below',
    ("My belief is that America has always been superior compared to others.",
      "The female nurse assisted the male doctor in the surgery",
        "He's an atheist, so he does not have any morals",
        "Our county boasts diverse landscapes and a vibrant community, offering opportunities and experiences that appeal to various interests and preferences."),
    index=None,
    placeholder="Select statement...",)

st.write('Your input:', text_to_classify_bert)

# Tokenize input text and make prediction
if st.button("Predict Bias",type="primary"):
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
