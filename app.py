from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
# importing classes from boolean.py
from boolean import TextCleaner, SpeechIndex, BoolSearch

app = FastAPI(title="Boolean IR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# load index at startup
print("loading index...")
cleaner = TextCleaner('Stopword-List.txt')
idx     = SpeechIndex(cleaner)

if os.path.exists('inverted_index.txt') and os.path.exists('positional_index.txt'):
    idx.loadIndexes()
else:
    idx.loadFolder('speeches')
    idx.dumpIndexes()

engine  = BoolSearch(idx, cleaner)
print("ready.\n")


class SearchReq(BaseModel):
    query: str
    type: str = "boolean"


@app.post("/search")
def search(req: SearchReq):
    q = req.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="query is empty")

    try:
        doc_ids = engine.search(q)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    results = []
    for did in doc_ids:
        fname = idx.docs.get(did, f'doc_{did}.txt')
        # pull out query terms to show as tags (skip operators / prox tokens)
        stems = []
        for tok in q.replace('(','').replace(')','').split():
            if tok.upper() in ('AND','OR','NOT') or tok.startswith('/'):
                continue
            s = cleaner.stemQ(tok)
            if s and s not in stems:
                stems.append(s)

        results.append({
            "doc_id":        did,
            "filename":      fname,
            "matched_terms": stems,
        })

    return {"results": results, "count": len(results)}


