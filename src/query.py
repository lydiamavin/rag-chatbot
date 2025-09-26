import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from src.vectorstore import load_faiss_index
from src.embed import load_chunks

def embed_query(query, model_name='all-MiniLM-L6-v2'):
    """
    Embed a query using Sentence Transformers.

    Args:
        query (str): The query text.
        model_name (str): Model name.

    Returns:
        np.ndarray: Query embedding.
    """
    model = SentenceTransformer(model_name)
    embedding = model.encode([query])
    return np.array(embedding)

def retrieve_top_k(query_embedding, index, chunks, k=5):
    """
    Retrieve top-k similar chunks from FAISS index.

    Args:
        query_embedding (np.ndarray): Query embedding.
        index: FAISS index.
        chunks (list): List of text chunks.
        k (int): Number of top results.

    Returns:
        list: List of top-k chunks.
    """
    # Normalize query embedding
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, k)
    top_chunks = [chunks[i] for i in indices[0]]
    return top_chunks

def generate_answer(query, context, model_name='google/flan-t5-small'):
    """
    Generate an answer using a text-to-text generation model.

    Args:
        query (str): The user's question.
        context (str): Retrieved context.
        model_name (str): Model name.

    Returns:
        str: Generated answer.
    """
    generator = pipeline('text2text-generation', model=model_name)
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    answer = generator(prompt, max_length=200, num_return_sequences=1)[0]['generated_text']
    return answer

def answer_question(query, index_path, chunks_json_path, k=5):
    """
    Answer a question using RAG.

    Args:
        query (str): User's question.
        index_path (str): Path to FAISS index.
        chunks_json_path (str): Path to chunks JSON.
        k (int): Number of top chunks to retrieve.

    Returns:
        tuple: (answer, top_chunks)
    """
    index = load_faiss_index(index_path)
    chunks = load_chunks(chunks_json_path)
    query_embedding = embed_query(query)
    top_chunks = retrieve_top_k(query_embedding, index, chunks, k)
    context = ' '.join(top_chunks)
    answer = generate_answer(query, context)
    return answer, top_chunks