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

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed" not in st.session_state:
    st.session_state.processed = False

# Sidebar for PDF upload and processing
with st.sidebar:
    st.header("Upload & Process PDFs")
    uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)

    if uploaded_files and not st.session_state.processed:
        st.success(f"Uploaded {len(uploaded_files)} PDF(s)")

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

                st.session_state.processed = True
                st.success("Processing complete! You can now chat.")

    if st.session_state.processed:
        st.info("PDFs processed. Ready to chat!")

# Main chat interface
st.header("Chat with your PDFs")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("Sources"):
                for i, source in enumerate(message["sources"]):
                    st.write(f"{i+1}. {source[:200]}...")

# Chat input
if prompt := st.chat_input("Ask a question about the uploaded PDFs"):
    if not st.session_state.processed:
        st.error("Please upload and process PDFs first.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                answer, sources = answer_question(prompt, "data/index.faiss", "data/combined_chunks.json")
                st.markdown(answer)
                with st.expander("Sources"):
                    for i, source in enumerate(sources):
                        st.write(f"{i+1}. {source[:200]}...")

        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})