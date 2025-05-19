# 📝 BART Text Summarizer

A simple Streamlit web app that uses a fine-tuned BART model to summarize long passages of text. Built using Hugging Face Transformers and PyTorch.

## 🚀 How to Run

1. Install dependencies:

   ```bash
   pip install streamlit torch transformers

2. Ensure the fine-tuned model directory bart-xsum-finetuned/ exists in the root folder and contains:
   
    config.json
   
    pytorch_model.bin
   
    tokenizer_config.json
   
    vocab.json
   
    merges.txt

3. Run the app:

   ```bash
   streamlit run app.py

# Output Sample



