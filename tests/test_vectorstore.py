import sys
sys.path.append('..')

import numpy as np
import pytest
from src.vectorstore import build_faiss_index, save_faiss_index, load_faiss_index
import tempfile
import os

def test_build_faiss_index():
    """Test building a FAISS index from embeddings."""
    # Create dummy embeddings
    embeddings = np.random.rand(10, 384).astype('float32')  # 384 is dimension for all-MiniLM-L6-v2
    index = build_faiss_index(embeddings)
    assert index is not None
    assert index.ntotal == 10

def test_save_and_load_faiss_index():
    """Test saving and loading FAISS index."""
    embeddings = np.random.rand(5, 384).astype('float32')
    index = build_faiss_index(embeddings)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".faiss") as tmp_file:
        tmp_path = tmp_file.name

    try:
        save_faiss_index(index, tmp_path)
        loaded_index = load_faiss_index(tmp_path)
        assert loaded_index.ntotal == 5
    finally:
        os.unlink(tmp_path)