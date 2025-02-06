import torch
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import streamlit as st
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model():
    model_name = "distilbert-base-uncased-distilled-squad"
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        model = DistilBertForQuestionAnswering.from_pretrained(model_name)
        logger.info("Model and tokenizer loaded successfully.")
        return tokenizer, model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None

def answer_question(tokenizer, model, context, question):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
    
    return answer.strip()

if __name__ == "__main__":
    st.title("SLM Question Answering System")
    tokenizer, model = load_model()
    
    context = st.text_area("Paste the book content here:")
    question = st.text_input("Enter your question:")
    
    if st.button("Get Answer"):
        if context and question:
            answer = answer_question(tokenizer, model, context, question)
            st.write("Answer:", answer if answer else "No answer found")
