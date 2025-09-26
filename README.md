# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that enables interactive conversations with PDF documents. Upload PDFs, process them, and ask questions in a chat interface powered by advanced NLP models.

## Features

- **PDF Ingestion**: Extract and chunk text from PDF documents using pdfplumber.
- **Embeddings**: Generate sentence embeddings with Sentence Transformers for semantic search.
- **Vector Search**: Efficient similarity search using FAISS vector database.
- **Question Answering**: Generate answers using pre-trained language models (e.g., FLAN-T5).
- **Chat Interface**: Interactive chat UI built with Streamlit for seamless user experience.
- **Session Management**: Persistent chat history and processed data per session.
- **Source Citations**: View relevant text chunks as sources for answers.

## Architecture

The application follows a modular RAG pipeline:

1. **Ingestion** (`src/ingest.py`): Extracts text from PDFs and chunks it into manageable pieces.
2. **Embedding** (`src/embed.py`): Converts text chunks into vector embeddings.
3. **Vector Store** (`src/vectorstore.py`): Builds and manages FAISS index for fast retrieval.
4. **Query** (`src/query.py`): Embeds user queries, retrieves relevant chunks, and generates answers.
5. **UI** (`app/streamlit_app.py`): Provides a chat interface for user interaction.

## Installation

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd rag-chatbot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app/streamlit_app.py
   ```

2. Open your browser to the provided URL (usually `http://localhost:8501`).

3. In the sidebar:
   - Upload one or more PDF files.
   - Click "Process PDFs" to ingest and index the documents.

4. Once processed, use the chat input at the bottom to ask questions about the PDFs.

5. View answers in the chat, with expandable sources for transparency.

### Example
- Upload a PDF about machine learning.
- Ask: "What is supervised learning?"
- Receive an answer with relevant excerpts from the document.

## Configuration

- **Models**: Default embedding model is `all-MiniLM-L6-v2`. Generation model is `google/flan-t5-small`. Modify in source code for customization.
- **Chunking**: Default chunk size is 1000 characters with 200 overlap. Adjust in `src/ingest.py`.
- **Data Storage**: Processed data (chunks, embeddings, index) is stored in the `data/` directory.

## Testing

Run tests with pytest:
```bash
pytest tests/
```

Current tests cover vectorstore functionality. Expand as needed.

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Make changes and add tests.
4. Run linting: `ruff check .` (if configured).
5. Commit and push: `git push origin feature-name`.
6. Open a pull request.

### Code Standards
- Use type hints where possible.
- Follow PEP 8 style.
- Add docstrings to functions.
- Ensure tests pass before submitting.

## License

[MIT License](LICENSE) - Feel free to use and modify.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/), [Sentence Transformers](https://www.sbert.net/), [FAISS](https://github.com/facebookresearch/faiss), and [Transformers](https://huggingface.co/docs/transformers/index).
