#more sentences per chunk -> faster the processing
#higher the batch size -> fater processing

import pandas as pd
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import subprocess

file_path = "D:/prog/DSC/Bjarne-Stroustrup-The-C-Plus-Plus-Programming-Language-4th-Edition.pdf"


def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def chunk_text(text, sentences_per_chunk=10):
    sentences = text.split(".")
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = ". ".join(sentences[i:i+sentences_per_chunk]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def build_faiss_index(chunks, batch_size=32):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    
    embeddings = embedder.encode(
        chunks,
        batch_size=batch_size
    )
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    
    df = pd.DataFrame({
        "chunk_id": range(len(chunks)),
        "text": chunks,
        "embedding": list(embeddings)
    })
    return embedder, index, df


def retrieve(query, embedder, index, df, k=3):
    query_vec = embedder.encode([query])
    D, I = index.search(np.array(query_vec), k)
    results = df.iloc[I[0]]
    return results["text"].tolist()


def ask_llama(prompt):
    result = subprocess.run(
        ["ollama", "run", "llama3.2:1b"],
        input=prompt.encode("utf-8"),
        capture_output=True,
    )
    return result.stdout.decode("utf-8")


def rag_query(query, embedder, index, df):
    context = "\n".join(retrieve(query, embedder, index, df))
    prompt = f"Use the following context to answer:\n{context}\nQuestion: {query}\nAnswer:"
    return ask_llama(prompt)


if __name__ == "__main__":
    print("Loading and processing PDF...")
    pdf_text = load_pdf(file_path)
    chunks = chunk_text(pdf_text, 12)
    embedder, index, df = build_faiss_index(chunks, batch_size=128)
    
    print(f"PDF loaded and indexed into {len(chunks)} chunks")
    
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        print("\n", rag_query(query, embedder, index, df))
