import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.title("📝 Free Shayari Generator (Open Source Model)")

keywords = st.text_input("Enter keywords for your Shayari (comma separated):", "")
recipient = st.selectbox("Who is the Shayari for?", ["Friend", "Lover", "Family", "Anyone"])
generate = st.button("Generate Shayari")

@st.cache_resource
def load_model():
    # You can switch to another available seq2seq or text-gen model of your choice!
    model_name = "google/mt5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

if generate and keywords.strip():
    with st.spinner("Generating Shayari..."):
        tokenizer, model = load_model()
        prompt = (
            f"Write a short Hindi Shayari for {recipient} with these words: {keywords}.\n"
            "Keep it poetic and meaningful."
        )
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        output_ids = model.generate(input_ids, max_length=64, temperature=0.9)
        shayari = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        st.subheader("✨ Your Shayari:")
        st.write(shayari)
else:
    st.info("Enter keywords and click Generate Shayari. No API key needed!")
