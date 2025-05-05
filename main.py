from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd

import os
from db_builder import build_chroma_db_if_needed


# Auto-build vector DB if not already built
build_chroma_db_if_needed()

# --- Setup ---
persist_directory = "./grocery_chroma_db"
client = chromadb.PersistentClient(path=persist_directory)
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-base-en-v1.5")
collection = client.get_or_create_collection(name="grocery_products", embedding_function=embed_fn)

# --- FastAPI Setup ---
app = FastAPI()

# CORS to allow frontend (e.g., Flutter Web) to talk to it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or use specific domains for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# --- Helper function for literal fallback search ---
def literal_fallback_search(query, query_tokens, all_docs):
    priority_matches = []
    normal_matches = []

    for name, meta in zip(all_docs['documents'], all_docs['metadatas']):
        name_lower = name.lower()
        if all(token in name_lower for token in query_tokens):
            match = {
                "name": name,
                "price": meta.get('price', 0.0),
                "store": meta.get('store', 'Unknown Store'),
                "brand": meta.get('brand', 'Unknown Brand')
            }
            if name_lower.startswith(query.strip().lower()):
                priority_matches.append(match)
            else:
                normal_matches.append(match)

    return priority_matches + normal_matches

# --- API Route ---
@app.get("/search")
def search_grocery_products(q: str = Query(..., min_length=1)):
    print(f"🔥 Received query: {q}")
    query_tokens = q.lower().split()

    # Literal search first
    all_docs = collection.get(include=["documents", "metadatas"])
    literal_results = literal_fallback_search(q, query_tokens, all_docs)

    # Semantic search if enough tokens
    semantic_results = []
    if len(query_tokens) >= 3:
        results = collection.query(
            query_texts=[q],
            n_results=20,
            include=["documents", "metadatas", "distances"]
        )
        if results and results.get('documents') and results['documents'][0]:
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]

            for doc, meta in zip(documents, metadatas):
                if not doc:
                    continue

                semantic_results.append({
                    "name": str(doc).strip(),
                    "price": meta.get('price', 'N/A'),
                    "store": meta.get('store', 'Unknown Store') if not pd.isna(meta.get('store')) else "Unknown Store",
                    "brand": meta.get('brand', 'Unknown Brand')
                })

    # Merge results: literal first, then semantic
    merged = literal_results + semantic_results
    return merged[:20]  # Limit to 20 results
