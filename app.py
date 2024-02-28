from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import streamlit as st

@st.cache_data
def download_model():
    model_name = 'facebook/mbart-large-50-many-to-many-mmt'
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    return model, tokenizer

st.title('English to Tamil Translator')
text = st.text_area("Enter Your Text:", value="", height=None, max_chars=None, key=None)
model, tokenizer = download_model()

if st.button('Translate to Tamil'):
    if text == '':
        st.write('Please Enter the Text')
    else:
        tokenizer.src_lang = "en_xx"
        encoded = tokenizer(text, return_tensors="pt")
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.lang_code_to_id["ta_IN"]
        )
        out = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        st.write('', str(out).strip('][\''))
else: 
    pass