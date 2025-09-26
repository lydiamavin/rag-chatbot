import json
import numpy as np
from sentence_transformers import SentenceTransformer

def load_chunks(json_path):
    """
    Load text chunks from a JSON file.

    Args:
        json_path (str): Path to the JSON file containing chunks.

    Returns:
        list: List of text chunks.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['chunks']

def embed_chunks(chunks, model_name='all-MiniLM-L6-v2'):
    """
    Embed the text chunks using Sentence Transformers.

    Args:
        chunks (list): List of text chunks.
        model_name (str): Name of the Sentence Transformer model.

    Returns:
        np.ndarray: Array of embeddings.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    return np.array(embeddings)

def save_embeddings(embeddings, output_path):
    """
    Save embeddings to a numpy file.

    Args:
        embeddings (np.ndarray): Array of embeddings.
        output_path (str): Path to save the embeddings.
    """
    np.save(output_path, embeddings)

def create_embeddings(json_path, embeddings_path):
    """
    Load chunks, embed them, and save embeddings.

    Args:
        json_path (str): Path to chunks JSON.
        embeddings_path (str): Path to save embeddings.npy.
    """
    chunks = load_chunks(json_path)
    embeddings = embed_chunks(chunks)
    save_embeddings(embeddings, embeddings_path)