import streamlit as st
from qa_bot import query_and_generate_answer

prompt = st.text_area("Enter your prompt:")

# Streamlit App
st.title("PDF Question Answering App")

if st.button("Get Answer"):
    if prompt:
        with st.spinner("Generating answer..."):
            answer = query_and_generate_answer(prompt)
            st.write("Answer:")
            st.write(answer)
    else:
        st.warning("Please enter a question.")