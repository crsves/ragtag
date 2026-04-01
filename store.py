"""
Step 4: Store embeddings with FAISS for fast similarity search
"""
import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Optional, Tuple


class VectorStore:
    """FAISS-based vector store for fast similarity search."""
    
    def __init__(self, embedding_dim: int):
        """
        Initialize vector store.
        
        Args:
            embedding_dim: Dimension of embeddings
        """
        self.embedding_dim = embedding_dim
        # Use IndexFlatIP for cosine similarity (requires normalized vectors)
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.metadata: List[Dict] = []
    
    def add(self, embeddings: np.ndarray, metadata: List[Dict]):
        """
        Add embeddings and metadata to the store.
        
        Args:
            embeddings: numpy array of shape (n, embedding_dim)
            metadata: List of metadata dicts for each embedding
        """
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings.astype('float32'))
        self.metadata.extend(metadata)
        
        print(f"Added {len(embeddings)} embeddings. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, List[Dict]]:
        """
        Search for k most similar chunks.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
        
        Returns:
            (scores, metadata) tuple
        """
        # Normalize query
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        # Get metadata for results
        results = [self.metadata[idx] for idx in indices[0]]
        
        return scores[0], results
    
    def save(self, output_dir: str):
        """Save index and metadata to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(output_path / 'faiss.index'))
        
        # Save metadata
        with open(output_path / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        
        # Save info
        info = {
            'embedding_dim': self.embedding_dim,
            'total_vectors': self.index.ntotal
        }
        with open(output_path / 'info.json', 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2)
        
        print(f"Saved vector store to {output_dir}")
    
    @classmethod
    def load(cls, store_dir: str) -> 'VectorStore':
        """Load vector store from disk."""
        store_path = Path(store_dir)
        
        # Load info
        with open(store_path / 'info.json', 'r') as f:
            info = json.load(f)
        
        # Create instance
        store = cls(info['embedding_dim'])
        
        # Load FAISS index
        store.index = faiss.read_index(str(store_path / 'faiss.index'))
        
        # Load metadata
        with open(store_path / 'metadata.json', 'r', encoding='utf-8') as f:
            store.metadata = json.load(f)
        
        print(f"Loaded vector store with {store.index.ntotal} vectors")
        return store


if __name__ == '__main__':
    # Load embeddings and metadata
    embeddings = np.load('processed/embeddings/embeddings.npy')
    with open('processed/embeddings/metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    print(f"Loaded {len(embeddings)} embeddings")

    # Create vector store
    store = VectorStore(embedding_dim=embeddings.shape[1])

    # Add in batches to avoid FAISS segfault on Apple Silicon with large datasets
    batch_size = 10000
    for i in range(0, len(embeddings), batch_size):
        batch_emb = embeddings[i:i + batch_size].copy()
        batch_meta = metadata[i:i + batch_size]
        store.add(batch_emb, batch_meta)

    # Save
    store.save('processed/vector_store')

    # Test loading
    loaded_store = VectorStore.load('processed/vector_store')
    print(f"Successfully loaded store with {loaded_store.index.ntotal} vectors")
