import faiss
import numpy as np

def build_faiss_index(embeddings):
    """
    Build a FAISS index from embeddings.

    Args:
        embeddings (np.ndarray): Array of embeddings.

    Returns:
        faiss.Index: FAISS index.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    # Normalize embeddings for cosine similarity
    embeddings = embeddings.astype('float32')
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index.add(embeddings)
    return index

def save_faiss_index(index, path):
    """
    Save FAISS index to file.

    Args:
        index (faiss.Index): FAISS index.
        path (str): Path to save the index.
    """
    faiss.write_index(index, path)

def load_faiss_index(path):
    """
    Load FAISS index from file.

    Args:
        path (str): Path to the index file.

    Returns:
        faiss.Index: Loaded FAISS index.
    """
    return faiss.read_index(path)

def create_and_save_index(embeddings_path, index_path):
    """
    Load embeddings, build index, and save it.

    Args:
        embeddings_path (str): Path to embeddings.npy.
        index_path (str): Path to save index.faiss.
    """
    embeddings = np.load(embeddings_path)
    index = build_faiss_index(embeddings)
    save_faiss_index(index, index_path)