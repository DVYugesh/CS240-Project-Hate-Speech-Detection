import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification,BertModel
from sklearn.metrics import accuracy_score, classification_report
import re
import os

# Set page title and layout
st.set_page_config(page_title="Hate Speech Detection Demo", layout="wide")

# Title and description
st.title("Hate Speech Detection in Portuguese")
st.markdown("""
This app demonstrates hate speech detection using six BERT-based models:
- **Large BERT (Binary)**: Classifies text as Hate or Non-Hate (`neuralmind/bert-large-portuguese-cased`).
- **Base BERT (Binary)**: Classifies text as Hate or Non-Hate (`neuralmind/bert-base-portuguese-cased`).
- **Hierarchical Large**: Categorizes hate speech into 11 types (ageism, aporophobia, etc.) using large bert
- **Hierarchical Base**: Categorizes hate speech into 11 types (ageism, aporophobia, etc.) using base bert
- **CNN Unfreezing Base BERT**: Binary classification with Base BERT + CNN, gradual unfreezing.
- **CNN Unfreezing Large BERT**: Binary classification with large BERT + CNN, gradual unfreezing.
            
Enter a text below to test the models or explore sample predictions.
""")

class BertWithCNN(nn.Module):
    def __init__(self, model_path, num_labels=2, freeze_bert=True):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.bert.config.hidden_size, out_channels=128, kernel_size=k)
            for k in [2, 3, 4]
        ])
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(128 * len(self.convs), num_labels)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state        # [B, L, H]
        x = x.permute(0, 2, 1)               # [B, H, L]

        x = [torch.relu(conv(x)) for conv in self.convs]
        x = [torch.max(conv, dim=2)[0] for conv in x]
        x = torch.cat(x, dim=1)              # [B, 128 * 3]

        x = self.dropout(x)
        logits = self.classifier(x)

        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
    

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# st.write(f"Using device: {device}")

# Define hate categories for hierarchical models
hate_labels = ['ageism', 'aporophobia', 'body_shame', 'capacitism', 'lgbtphobia',
               'political', 'racism', 'religious_intolerance', 'misogyny', 'xenophobia', 'other']

# Load models and tokenizers
@st.cache_resource
def load_model_and_tokenizer(path, model_type="bert", num_labels=2, problem_type="single_label_classification"):
    try:
        # Verify tokenizer files
        if not os.path.exists(path) or not os.path.exists(os.path.join(path, "vocab.txt")):
            raise FileNotFoundError(f"Tokenizer files missing in {path}")
        tokenizer = BertTokenizer.from_pretrained(path)
        
        if model_type == "bert":
            model = BertForSequenceClassification.from_pretrained(
                path, num_labels=num_labels, problem_type=problem_type
            )
        elif model_type == "cnn":
            # Load the full model from .pth file
            if path == "./Improved_models/CNN_grad_unfreezing_bert_base_binary":
                model_path = f"{path}/CNN_grad_unfreeze_bert.pth"
            elif path == "./Improved_models/CNN_grad_unfreezing_bert_large_binary":
                model_path = f"{path}/CNN_grad_unfreeze_bertl.pth"
            else:
                raise ValueError(f"Unknown CNN model path: {path}")
            
            # Verify .pth file exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file {model_path} not found")
            
            # Load model with binary mode explicitly
            try:
                model = torch.load(model_path, map_location=device,weights_only=False)
            except Exception as e:
                raise RuntimeError(f"Failed to load .pth file from {model_path}: {str(e)}")
        else:
            raise ValueError("Invalid model_type")
        
        model.to(device)
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model from {path}: {str(e)}")
        raise

# Load all six models
try:
    binary_large_model, binary_large_tokenizer = load_model_and_tokenizer("./Bertimbau_models/Bert-large-binary")
    binary_small_model, binary_small_tokenizer = load_model_and_tokenizer("./Bertimbau_models/Bert-base-binary")
    hier_large_model, hier_large_tokenizer = load_model_and_tokenizer(
        "./Bertimbau_models/Bert-large-cat", num_labels=len(hate_labels), problem_type="multi_label_classification"
    )
    hier_small_model, hier_small_tokenizer = load_model_and_tokenizer(
        "./Bertimbau_models/Bert-base-cat", num_labels=len(hate_labels), problem_type="multi_label_classification"
    )
    cnn_unfreeze_small_model, cnn_unfreeze_small_tokenizer = load_model_and_tokenizer(
        "./Improved_models/CNN_grad_unfreezing_bert_base_binary", model_type="cnn", num_labels=2
    )
    cnn_unfreeze_large_model, cnn_unfreeze_large_tokenizer = load_model_and_tokenizer(
        "./Improved_models/CNN_grad_unfreezing_bert_large_binary", model_type="cnn", num_labels=2
    )
except Exception as e:
    st.error(f"Failed to load models: {str(e)}")
    st.stop()

# Preprocess text function
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Predict function
def predict_text(text, model_type):
    clean_text = preprocess_text(text)
    
    # Select model and tokenizer
    if model_type == "binary_large":
        model, tokenizer = binary_large_model, binary_large_tokenizer
    elif model_type == "binary_small":
        model, tokenizer = binary_small_model, binary_small_tokenizer
    elif model_type == "hier_large":
        model, tokenizer = hier_large_model, hier_large_tokenizer
    elif model_type == "hier_small":
        model, tokenizer = hier_small_model, hier_small_tokenizer
    elif model_type == "cnn_unfreeze_small":
        model, tokenizer = cnn_unfreeze_small_model, cnn_unfreeze_small_tokenizer
    elif model_type == "cnn_unfreeze_large":
        model, tokenizer = cnn_unfreeze_large_model, cnn_unfreeze_large_tokenizer
    else:
        raise ValueError("Invalid model_type")

    # Tokenize
    inputs = tokenizer(clean_text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Predict
    with torch.no_grad():
        if model_type in ["cnn_unfreeze_small", "cnn_unfreeze_large"]:
            # CNN models: Pass only input_ids and attention_mask
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        else:
            # BERT models: Pass all inputs (including token_type_ids)
            outputs = model(**inputs)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        
        if model_type in ["binary_large", "binary_small"]:
            # Binary classification (BERT models)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            predicted_class = np.argmax(probs)
            return {"label": "Hate" if predicted_class == 1 else "Non-Hate", "probs": probs}
        elif model_type in ["cnn_unfreeze_small", "cnn_unfreeze_large"]:
            # Binary classification (CNN models)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            predicted_class = np.argmax(probs)
            return {"label": "Hate" if predicted_class == 1 else "Non-Hate", "probs": probs}
        else:
            # Multilabel classification
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            predicted_labels = [hate_labels[i] for i in range(len(hate_labels)) if probs[i] >= 0.5]
            non_hate = len(predicted_labels) == 0
            return {
                "labels": ["non_hate"] if non_hate else predicted_labels,
                "probs": dict(zip(hate_labels + ["non_hate"], list(probs) + [float(non_hate)]))
            }

# Streamlit layout
st.header("Test Your Text")
user_input = st.text_area("Enter text in Portuguese:", height=100)

# Create two sections of buttons
st.subheader("Binary Classification")
# First row: Large BERT, Small BERT
col1, col2 = st.columns(2)
with col1:
    if st.button("Classify Using Large BERT"):
        if user_input.strip():
            result = predict_text(user_input, "binary_large")
            st.success(f"Large BERT Prediction: **{result['label']}**")
            st.write(f"Confidence: Non-Hate: {result['probs'][0]:.4f}, Hate: {result['probs'][1]:.4f}")
        else:
            st.error("Please enter some text.")
with col2:
    if st.button("Classify Using Base BERT"):
        if user_input.strip():
            result = predict_text(user_input, "binary_small")
            st.success(f"Base BERT Prediction: **{result['label']}**")
            st.write(f"Confidence: Non-Hate: {result['probs'][0]:.4f}, Hate: {result['probs'][1]:.4f}")
        else:
            st.error("Please enter some text.")

# Second row: CNN Unfreezing Small BERT, CNN Unfreezing Large BERT
col3, col4 = st.columns(2)
with col3:
    if st.button("Classify Using CNN Unfreezing Base BERT"):
        if user_input.strip():
            result = predict_text(user_input, "cnn_unfreeze_small")
            st.success(f"CNN Unfreezing Base BERT Prediction: **{result['label']}**")
            st.write(f"Confidence: Non-Hate: {result['probs'][0]:.4f}, Hate: {result['probs'][1]:.4f}")
        else:
            st.error("Please enter some text.")
with col4:
    if st.button("Classify Using CNN Unfreezing Large BERT"):
        if user_input.strip():
            result = predict_text(user_input, "cnn_unfreeze_large")
            st.success(f"CNN Unfreezing Large BERT Prediction: **{result['label']}**")
            st.write(f"Confidence: Non-Hate: {result['probs'][0]:.4f}, Hate: {result['probs'][1]:.4f}")
        else:
            st.error("Please enter some text.")

st.subheader("Hierarchical Classification")
col5, col6 = st.columns(2)
with col5:
    if st.button("Classify Using Hierarchical Large"):
        if user_input.strip():
            result = predict_text(user_input, "hier_large")
            st.success(f"Hierarchical Large Prediction: **{', '.join(result['labels'])}**")
            with st.expander("Show Confidence Scores"):
                st.write("Confidence Scores:")
                for label, prob in result['probs'].items():
                    st.write(f"{label}: {prob:.4f}")
        else:
            st.error("Please enter some text.")
with col6:
    if st.button("Classify Using Hierarchical Base"):
        if user_input.strip():
            result = predict_text(user_input, "hier_small")
            st.success(f"Hierarchical Base Prediction: **{', '.join(result['labels'])}**")
            with st.expander("Show Confidence Scores"):
                st.write("Confidence Scores:")
                for label, prob in result['probs'].items():
                    st.write(f"{label}: {prob:.4f}")
        else:
            st.error("Please enter some text.")

# Optional: Display sample predictions
st.header("Sample Predictions")
if st.checkbox("Show sample predictions"):
    sample_texts = [
        "Você é incrível, continue assim!",  # Non-Hate
        "Ninguém gosta de você, seu idiota!",  # Hate (possibly body_shame, misogyny)
        "O dia está lindo hoje!",  # Non-Hate
        "Estrangeiros estão roubando nossos empregos!",  # Hate (xenophobia)
    ]
    sample_results = []
    for text in sample_texts:
        binary_large = predict_text(text, "binary_large")
        binary_small = predict_text(text, "binary_small")
        hier_large = predict_text(text, "hier_large")
        hier_small = predict_text(text, "hier_small")
        cnn_unfreeze_small = predict_text(text, "cnn_unfreeze_small")
        cnn_unfreeze_large = predict_text(text, "cnn_unfreeze_large")
        sample_results.append({
            "Text": text,
            "Binary Large": binary_large["label"],
            "Binary Base": binary_small["label"],
            "Hierarchical Large": ", ".join(hier_large["labels"]),
            "Hierarchical Base": ", ".join(hier_small["labels"]),
            "CNN Unfreezing Base": cnn_unfreeze_small["label"],
            "CNN Unfreezing Large": cnn_unfreeze_large["label"]
        })
    st.dataframe(pd.DataFrame(sample_results))

