import streamlit as st
from transformers import BertTokenizer, BertForMaskedLM
import torch

@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.eval()
    return tokenizer, model


def predict_masked_word(text, tokenizer, model):
    # Tokenize input with mask token
    inputs = tokenizer(text, return_tensors='pt')
    mask_token_index = torch.where(inputs.input_ids == tokenizer.mask_token_id)[1]

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    mask_token_logits = logits[0, mask_token_index, :]
    top_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

    results = []
    for token in top_tokens:
        word = tokenizer.decode([token]).strip()
        results.append(word)
    return results

def main():
    st.title("BERT-base-uncased Masked Word Prediction")

    st.write("Enter a sentence with one [MASK] token, e.g.:")
    st.write("`The man worked as a [MASK].`")

    user_input = st.text_input("Input text", value="The man worked as a [MASK].")

    if "[MASK]" not in user_input:
        st.warning("Please include one [MASK] token in your input.")
        return

    tokenizer, model = load_model()

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            predictions = predict_masked_word(user_input, tokenizer, model)
        st.write("Top predictions for the [MASK]:")
        for i, word in enumerate(predictions):
            st.write(f"{i+1}. {word}")

if __name__ == "__main__":
    main()
