import pdfplumber
import json
import os

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file using pdfplumber.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Chunk the text into smaller pieces with overlap.

    Args:
        text (str): The full text to chunk.
        chunk_size (int): Size of each chunk in characters.
        overlap (int): Number of characters to overlap between chunks.

    Returns:
        list: List of text chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start >= len(text):
            break
    return chunks

def save_chunks_to_json(chunks, output_path):
    """
    Save the list of chunks to a JSON file.

    Args:
        chunks (list): List of text chunks.
        output_path (str): Path to save the JSON file.
    """
    data = {"chunks": chunks}
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

def ingest_pdf(pdf_path, output_json_path):
    """
    Ingest a PDF file: extract text, chunk it, and save to JSON.

    Args:
        pdf_path (str): Path to the PDF file.
        output_json_path (str): Path to save the chunks JSON.
    """
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    save_chunks_to_json(chunks, output_json_path)