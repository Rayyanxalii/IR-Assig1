# Boolean Information Retrieval System
### IR Assignment 1 — Trump Speeches (56 documents)

A Boolean Information Retrieval system built in Python that supports **AND**, **OR**, **NOT**, and **proximity** queries over 56 Trump speech documents.

---

## Files

| File | Description |
|------|-------------|
| `boolean.py` | Core IR system — text preprocessing (tokenization, stopword removal, Porter stemming), inverted index construction, positional index construction, and Boolean/proximity query processing. Run this directly for a CLI search interface. |
| `app.py` | FastAPI backend — exposes a `POST /search` endpoint that the frontend calls. Loads the index on startup (from saved files if available, otherwise builds from speeches). |
| `index.html` | Frontend UI — single-page search interface built with HTML, Tailwind CSS, and vanilla JavaScript. Sends queries to the FastAPI backend and displays results. |
| `inverted_index.txt` | Auto-generated inverted index saved to disk. Format: `term -> [doc_id1, doc_id2, ...]` |
| `positional_index.txt` | Auto-generated positional index saved to disk. Format: `term -> {doc_id: [pos1, pos2, ...], ...}` |
| `Stopword-List.txt` | List of stopwords used during preprocessing. Stopword positions are still counted for proximity queries. |
| `speeches/` | Folder containing the 56 Trump speech `.txt` files used to build the index. |

---

## How to Run

### Option 1 — CLI (terminal only)
```bash
py boolean.py
```
Builds the index, saves it to txt files, and opens an interactive query prompt.

### Option 2 — Web UI (with FastAPI backend)
```bash
# Install dependencies (first time only)
pip install fastapi uvicorn nltk

# Start the backend server
python -m uvicorn app:app --host 127.0.0.1 --port 5000

# Then open index.html in your browser
```

---

## Query Syntax

| Type | Example | Description |
|------|---------|-------------|
| Boolean AND | `trump AND america` | Documents containing both terms |
| Boolean OR | `clinton OR trump` | Documents containing either term |
| Boolean NOT | `trump NOT mexico` | Documents with trump but not mexico |
| Combined | `trump AND wall NOT clinton` | Chained boolean operators |
| Proximity | `trump america /3` | trump appears exactly 3 words before america |
| Phrase | `donald trump` | Two words treated as adjacent phrase (gap = 0) |

---

## Index Details

- **Documents:** 56 Trump speeches
- **Preprocessing:** Lowercase → punctuation removal → stopword filtering → Porter stemming
- **Inverted Index:** Maps each stemmed term to the set of documents it appears in
- **Positional Index:** Maps each stemmed term to per-document lists of raw token positions (including stopword positions, required for accurate proximity queries)
- **Indexes are saved** to `inverted_index.txt` and `positional_index.txt` on first run and **loaded from disk** on subsequent runs for faster startup
