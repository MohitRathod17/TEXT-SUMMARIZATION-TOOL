import streamlit as st
st.set_page_config(page_title="BART Text Summarizer", layout="centered")

import torch
from transformers import BartForConditionalGeneration, BartTokenizer

# Load your fine-tuned model and tokenizer from saved directory
@st.cache_resource
def load_model():
    save_directory = "./bart-xsum-finetuned"
    tokenizer = BartTokenizer.from_pretrained(save_directory)
    model = BartForConditionalGeneration.from_pretrained(save_directory)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# Streamlit UI
st.title("Text Summarization with Fine-Tuned BART")
st.markdown("This app uses a fine-tuned BART model to generate summaries of long text passages.")

text_input = st.text_area("Enter the article or document text below:", height=300)

if st.button("Generate Summary"):
    if text_input.strip() == "":
        st.warning("Please enter some text to summarize.")
    else:
        with st.spinner("Generating summary..."):
            # Tokenize input
            inputs = tokenizer([text_input], max_length=1024, return_tensors="pt", truncation=True).to(device)

            # Generate summary
            summary_ids = model.generate(
                inputs["input_ids"],
                num_beams=4,
                max_length=150,
                early_stopping=True
            )

            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        st.subheader("Generated Summary")
        st.success(summary)





