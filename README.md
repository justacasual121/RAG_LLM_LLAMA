
# PDF RAG Pipeline with LLaMA

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that allows you to query PDF documents using a **LLaMA** language model. It extracts text from PDFs, embeds the text for retrieval, and uses context-aware generation to answer user questions.

A **sample PDF** (`Bjarne-Stroustrup-The-C-Plus-Plus-Programming-Language-4th-Edition.pdf`) has been included for testing, which is a C++ programming book.

---
---

## Features

- Extract text from any PDF file.
- Split text into chunks for embedding and retrieval.
- Use **SentenceTransformers** for high-quality semantic embeddings.
- Store embeddings in **FAISS** for fast similarity search.
- Query **LLaMA 3.2 1B** for answers based on retrieved chunks.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/justacasual121/RAG_LLM_LLAMA.git
cd RAG_LLM_LLAMA
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

1. Update the `file_path` in `pdf_rag.py` to point to your PDF file:

```python
file_path = "D:/path/to/your/file.pdf"
```

2. Run the script:

```bash
Rag_LLM.py
```

3. Interactively ask questions in the terminal. Type `exit` to quit.

---

## Configuration Notes

### 1. Chunking (`sentences_per_chunk`)
- `chunk_text()` splits the PDF into chunks of sentences.
- **More sentences per chunk** → fewer chunks → faster FAISS indexing and retrieval, but less granular retrieval.
- **Fewer sentences per chunk** → more chunks → finer retrieval but slower processing.

Example:

```python
chunks = chunk_text(pdf_text, sentences_per_chunk=12)
```

---

### 2. FAISS Embedding Batch Size (`batch_size`)
- `build_faiss_index()` generates embeddings in batches.
- **Higher batch size** → faster embedding generation (if enough RAM), but uses more memory.
- **Lower batch size** → slower but more memory-efficient.

Example:

```python
embedder, index, df = build_faiss_index(chunks, batch_size=128)
```

---

### 3. Sentence Transformer Model
- Current model: `all-distilroberta-v1`
- **Other options:**
  - `all-MiniLM-L6-v2` → faster, smaller, CPU-friendly.
  - `all-mpnet-base-v2` → higher accuracy for retrieval tasks.
- Change the model in `build_faiss_index()`:

```python
embedder = SentenceTransformer("all-distilroberta-v1")
```

---

### 4. LLaMA Model
- Current model: `llama3.2:1b` via Ollama.
- You can adjust the number of tokens generated per answer by using `/set num_predict` in the prompt or globally in Ollama.
- Example in Python:

```python
full_input = f"/set num_predict 400\n{prompt}"
```

---

## Performance Tips
1. **Faster processing:** Increase `sentences_per_chunk` and `batch_size`.  
2. **Memory efficiency:** Reduce `batch_size`.  
3. **Longer answers:** Increase LLaMA `num_predict` tokens.  
4. **Caching:** Consider saving FAISS index and embeddings to avoid reprocessing PDFs each run.

---

## Dependencies

- `pandas` – for managing chunks and embeddings.
- `PyPDF2` – PDF text extraction.
- `sentence-transformers` – generating semantic embeddings.
- `faiss-cpu` – similarity search.
- `numpy` – numerical operations.
- `subprocess` – calling LLaMA model via Ollama.

---

