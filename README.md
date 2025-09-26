# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents, ingest them, and ask questions based on the content using a Streamlit UI.

## Features

- PDF document ingestion and text chunking
- Sentence embeddings using Sentence Transformers
- FAISS vector store for efficient similarity search
- Question answering with a pre-trained language model
- Streamlit web interface for easy interaction

## Quickstart

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd rag-chatbot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app/streamlit_app.py
   ```

4. Open your browser to the provided URL and start chatting with your PDFs!
