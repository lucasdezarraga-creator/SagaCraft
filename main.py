import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb

app = FastAPI()

# Enable CORS for Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Initialize local persistent ChromaDB (Saves to a local folder!)
chroma_client = chromadb.PersistentClient(path="./lore_db")

# Collection A: Stores core entity profiles (e.g., "Elion - lost his left arm in Ch. 3")
character_profiles = chroma_client.get_or_create_collection(name="character_profiles")

# Collection B: Stores chapter events & raw dialogue
chapter_logs = chroma_client.get_or_create_collection(name="chapter_logs")


class SeedLoreRequest(BaseModel):
    character_id: str
    summary: str

class SceneDraftRequest(BaseModel):
    draft_text: str

@app.post("/api/seed-character")
async def seed_character(data: SeedLoreRequest):
    """Seed or update a character's established lore profile."""
    character_profiles.upsert(
        ids=[data.character_id],
        documents=[data.summary],
        metadatas=[{"type": "character_summary"}]
    )
    return {"status": "success", "message": f"Updated profile for {data.character_id}"}


@app.post("/api/check-consistency")
async def check_consistency(data: SceneDraftRequest):
    """
    RAG Step: Given a new draft scene, retrieve relevant character profiles 
    and event logs, then check for logical lore conflicts.
    """
    try:
        # 1. Retrieve top 2 matching character traits/summaries
        profile_matches = character_profiles.query(
            query_texts=[data.draft_text],
            n_results=2
        )
        
        retrieved_lore = []
        if profile_matches["documents"]:
            for docs in profile_matches["documents"]:
                retrieved_lore.extend(docs)

        # Build context string
        context_str = "\n".join(retrieved_lore) if retrieved_lore else "No matching lore profiles found."

        # 2. Return retrieved lore context + draft back to UI (or evaluate via LLM)
        return {
            "draft_analyzed": data.draft_text,
            "retrieved_established_lore": context_str,
            "has_conflict": False, # We can pass this context to an LLM to evaluate True/False!
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)