import streamlit as st
import os
from utils.pdf_utils import extract_text_from_pdf, chunk_text
from utils.embed_utils import embed_chunks, build_faiss_index, search_faiss
from utils.llm_utils import ask_llm
import numpy as np

st.set_page_config("ChatwithPDFs")
st.title("Chat with your PDF")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    save_dir = "data"
    os.makedirs(save_dir, exist_ok=True)  # <-- Add this line
    pdf_path = os.path.join(save_dir, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    index = build_faiss_index(np.array(embeddings))

    st.session_state.chunks = chunks
    st.session_state.embeddings = embeddings
    st.session_state.index = index

if "index" in st.session_state:
    question = st.text_input("Ask a question about your PDF:")
    if question:
        query_embedding = embed_chunks([question])
        idxs = search_faiss(st.session_state.index, np.array(query_embedding))
        context = " ".join([st.session_state.chunks[i] for i in idxs])
        answer = ask_llm(question, context)
        st.write(answer)
