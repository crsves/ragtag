"""
Step 3: Generate embeddings using local model
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class EmbeddingGenerator:
    """Generate embeddings using local MiniLM model."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize embedding model.
        
        Args:
            model_name: HuggingFace model name (all-MiniLM-L6-v2 is fast and CPU-friendly)
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        print("Model loaded successfully")
    
    def embed_chunks(self, chunks: List[Dict], batch_size: int = 16,
                     progress_cb=None, pct_start: int = 40, pct_end: int = 90) -> np.ndarray:
        """
        Generate embeddings for all chunks.

        Args:
            chunks: List of chunk dicts
            batch_size: Process this many chunks at once
            progress_cb: Optional callable(pct: int, msg: str) for per-batch progress
            pct_start: progress percentage to start from (for the caller's range)
            pct_end: progress percentage to end at
        Returns:
            numpy array of embeddings (n_chunks, embedding_dim)
        """
        texts = [chunk['text'] for chunk in chunks]
        total = len(texts)
        print(f"Generating embeddings for {total} chunks...")

        if progress_cb is None or total == 0:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
            return embeddings

        # Split into batches and report per-batch progress.
        batches = [texts[i:i + batch_size] for i in range(0, total, batch_size)]
        all_embeddings = []
        pct_range = pct_end - pct_start
        for idx, batch in enumerate(batches):
            batch_embeddings = self.model.encode(
                batch,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            all_embeddings.append(batch_embeddings)
            done = (idx + 1) * batch_size
            pct = pct_start + int((idx + 1) / len(batches) * pct_range)
            progress_cb(pct, f"Embedding… batch {idx+1}/{len(batches)} ({min(done, total)}/{total} chunks)")

        return np.concatenate(all_embeddings, axis=0)
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text
        
        Returns:
            numpy array of shape (embedding_dim,)
        """
        return self.model.encode(query, convert_to_numpy=True)


def save_embeddings(embeddings: np.ndarray, chunks: List[Dict], output_dir: str):
    """
    Save embeddings and metadata.
    
    Args:
        embeddings: numpy array of embeddings
        chunks: List of chunk metadata
        output_dir: Directory to save to
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save embeddings as numpy array
    np.save(output_path / 'embeddings.npy', embeddings)
    print(f"Saved embeddings to {output_path / 'embeddings.npy'}")
    
    # Save metadata
    with open(output_path / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata to {output_path / 'metadata.json'}")
    
    # Save info
    info = {
        'n_chunks': len(chunks),
        'embedding_dim': embeddings.shape[1],
        'model_name': 'all-MiniLM-L6-v2'
    }
    with open(output_path / 'info.json', 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)
    print(f"Saved info to {output_path / 'info.json'}")


if __name__ == '__main__':
    # Load chunks
    with open('processed/chunks.json', 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks")
    
    # Generate embeddings
    generator = EmbeddingGenerator()
    embeddings = generator.embed_chunks(chunks)
    
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Save
    save_embeddings(embeddings, chunks, 'processed/embeddings')
