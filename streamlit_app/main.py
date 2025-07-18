import streamlit as st
from app.ingest import extract_text_from_pdf, chunk_text
from app.embedder import get_embeddings
from app.vector_store import FAISSStore
from app.qa_engine import answer_question
from sentence_transformers import SentenceTransformer

st.set_page_config(layout="wide")
st.title("📘 Ask Me Anything - Long Document Q&A")

uploaded_file = st.file_uploader("Upload a long PDF", type="pdf")

if uploaded_file:
    st.info("Extracting text...")
    text = extract_text_from_pdf(uploaded_file)
    chunks = chunk_text(text)
    
    st.success(f"Document chunked into {len(chunks)} sections.")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = get_embeddings(chunks)
    
    store = FAISSStore(dim=384)
    store.add(embeddings, chunks)

    question = st.text_input("Ask a question about the document")
    if st.button("Get Answer") and question:
        q_embedding = model.encode([question])[0]
        top_chunks = store.search(q_embedding)
        answer = answer_question(top_chunks, question)
        st.subheader("Answer:")
        st.write(answer)
