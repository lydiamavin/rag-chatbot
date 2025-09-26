import streamlit as st
import os
import sys
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ingest import ingest_pdf
from src.embed import create_embeddings
from src.vectorstore import create_and_save_index
from src.query import answer_question

st.title("RAG Chatbot")

# File uploader for PDFs
uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    st.success(f"Uploaded {len(uploaded_files)} PDF(s)")

    # Process button
    if st.button("Process PDFs"):
        with st.spinner("Processing PDFs..."):
            # Create data directory if not exists
            os.makedirs("data", exist_ok=True)

            all_text = ""
            for uploaded_file in uploaded_files:
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                # Ingest
                json_path = f"data/chunks_{uploaded_file.name}.json"
                ingest_pdf(tmp_path, json_path)

                # Load chunks and append text
                from src.embed import load_chunks
                chunks = load_chunks(json_path)
                all_text += " ".join(chunks) + " "

                # Clean up temp
                os.unlink(tmp_path)

            # Now, chunk the combined text
            from src.ingest import chunk_text, save_chunks_to_json
            combined_chunks = chunk_text(all_text)
            combined_json = "data/combined_chunks.json"
            save_chunks_to_json(combined_chunks, combined_json)

            # Embed
            embeddings_path = "data/embeddings.npy"
            create_embeddings(combined_json, embeddings_path)

            # Build index
            index_path = "data/index.faiss"
            create_and_save_index(embeddings_path, index_path)

            st.success("Processing complete! You can now ask questions.")

# Question input
question = st.text_input("Ask a question about the uploaded PDFs")

if st.button("Ask"):
    if not os.path.exists("data/index.faiss"):
        st.error("Please upload and process PDFs first.")
    else:
        with st.spinner("Generating answer..."):
            answer, sources = answer_question(question, "data/index.faiss", "data/combined_chunks.json")
            st.write("**Answer:**", answer)
            st.write("**Sources:**")
            for i, source in enumerate(sources):
                st.write(f"{i+1}. {source[:200]}...")  # Show first 200 chars