from typing import List, Dict, Any
import numpy as np
from pathlib import Path
import json
import os
from openai import OpenAI
import faiss
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

class ConversationEmbeddings:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embeddings_dir = Path("./data") / str(user_id) / "embeddings"
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.embeddings_dir / "faiss_index.bin"
        self.metadata_path = self.embeddings_dir / "metadata.json"
        self.dimension = 1536  # OpenAI ada-002 embedding dimension
        
        # Initialize or load FAISS index
        if self.index_path.exists() and self.metadata_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = []
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text using OpenAI's API"""
        response = self.openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    
    def add_conversation(self, question: str, answer: str, timestamp: str = None) -> None:
        """Add a conversation to the vector store"""
        if timestamp is None:
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        
        # Combine question and answer for context
        full_text = f"Question: {question}\nAnswer: {answer}"
        
        # Get embedding
        embedding = self._get_embedding(full_text)
        
        # Add to FAISS index
        self.index.add(embedding.reshape(1, -1))
        
        # Add metadata
        self.metadata.append({
            "question": question,
            "answer": answer,
            "timestamp": timestamp,
            "index": len(self.metadata)
        })
        
        # Save updated index and metadata
        self._save_state()
    
    def search_conversations(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar conversations using the query"""
        if len(self.metadata) == 0:
            return []
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        
        # Get results with metadata
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):  # Ensure valid index
                result = self.metadata[idx].copy()
                result["distance"] = float(distances[0][i])
                results.append(result)
        
        return results
    
    def _save_state(self) -> None:
        """Save the current state of the index and metadata"""
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def clear(self) -> None:
        """Clear all embeddings and metadata"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        self._save_state() 