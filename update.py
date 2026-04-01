"""
Step 8: Update system with new messages incrementally
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
from normalize import normalize_messages
from chunk import chunk_messages
from embed import EmbeddingGenerator
from store import VectorStore


class RAGUpdater:
    """Incrementally update the RAG system with new messages."""
    
    def __init__(self, 
                 store_dir: str = 'processed/vector_store',
                 model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize updater."""
        self.store_dir = store_dir
        self.embedding_generator = EmbeddingGenerator(model_name)
        
        # Load existing store
        try:
            self.vector_store = VectorStore.load(store_dir)
            self.existing_count = self.vector_store.index.ntotal
            print(f"Loaded existing store with {self.existing_count} chunks")
        except Exception as e:
            print(f"Could not load existing store: {e}")
            print("Will create new store after processing new messages")
            self.vector_store = None
            self.existing_count = 0
    
    def update_from_new_file(self, new_file_path: str, messages_per_chunk: int = 1):
        """
        Process new messages from a file and add to index.
        
        Args:
            new_file_path: Path to new messages JSON file
            messages_per_chunk: Chunking strategy (same as original)
        """
        print(f"\n{'='*80}")
        print("Processing new messages...")
        print('='*80)
        
        # Step 1: Normalize new messages
        print("\n[1/4] Normalizing new messages...")
        normalized = self._normalize_new_messages(new_file_path)
        
        if not normalized:
            print("No new messages to process")
            return
        
        # Step 2: Chunk new messages
        print(f"\n[2/4] Chunking {len(normalized)} messages...")
        chunks = chunk_messages(normalized, messages_per_chunk=messages_per_chunk)
        
        # Update chunk IDs to start after existing chunks
        for chunk in chunks:
            chunk['chunk_id'] = self.existing_count + chunk['chunk_id']
        
        print(f"Created {len(chunks)} new chunks")
        
        # Step 3: Generate embeddings
        print(f"\n[3/4] Generating embeddings for new chunks...")
        embeddings = self.embedding_generator.embed_chunks(chunks)
        
        # Step 4: Add to vector store
        print(f"\n[4/4] Adding to vector store...")
        if self.vector_store is None:
            # Create new store
            self.vector_store = VectorStore(embedding_dim=embeddings.shape[1])
        
        self.vector_store.add(embeddings, chunks)
        
        # Save updated store
        self.vector_store.save(self.store_dir)
        
        print(f"\n{'='*80}")
        print(f"Successfully added {len(chunks)} new chunks!")
        print(f"Total chunks in store: {self.vector_store.index.ntotal}")
        print('='*80)
    
    def _normalize_new_messages(self, file_path: str) -> List[Dict]:
        """Normalize new messages from file."""
        temp_output = 'processed/temp_normalized.json'
        normalized = normalize_messages(file_path, temp_output)
        
        # Clean up temp file
        Path(temp_output).unlink(missing_ok=True)
        
        return normalized
    
    def update_from_normalized(self, normalized_messages: List[Dict], 
                               messages_per_chunk: int = 1):
        """
        Update from already-normalized messages.
        
        Useful if you have an incremental export or want to process
        only messages after a certain date.
        """
        print(f"Processing {len(normalized_messages)} normalized messages...")
        
        # Chunk
        chunks = chunk_messages(normalized_messages, messages_per_chunk=messages_per_chunk)
        
        # Update IDs
        for chunk in chunks:
            chunk['chunk_id'] = self.existing_count + chunk['chunk_id']
        
        # Generate embeddings
        embeddings = self.embedding_generator.embed_chunks(chunks)
        
        # Add to store
        if self.vector_store is None:
            self.vector_store = VectorStore(embedding_dim=embeddings.shape[1])
        
        self.vector_store.add(embeddings, chunks)
        self.vector_store.save(self.store_dir)
        
        print(f"Added {len(chunks)} new chunks. Total: {self.vector_store.index.ntotal}")


def get_latest_timestamp(store_dir: str = 'processed/vector_store') -> str:
    """Return the latest timestamp_end across all chunks in the store."""
    metadata_path = Path(store_dir) / 'metadata.json'
    if not metadata_path.exists():
        return None
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    timestamps = [m['timestamp_end'] for m in metadata if m.get('timestamp_end')]
    if not timestamps:
        return None
    return max(timestamps)


if __name__ == '__main__':
    import sys
    from datetime import datetime, timezone

    args = sys.argv[1:]

    # Parse optional --store-dir <path>
    store_dir = 'processed/vector_store'
    if '--store-dir' in args:
        idx = args.index('--store-dir')
        store_dir = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    if not args or args[0] == '--check':
        latest = get_latest_timestamp(store_dir)
        if latest is None:
            print("No existing data found in vector store.")
            sys.exit(1)
        dt = datetime.fromisoformat(latest)
        dt_utc = dt.astimezone(timezone.utc)
        now_utc = datetime.now(timezone.utc)
        print(f"Latest message in store : {dt_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Current time            : {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print()
        print("Export range to use in DiscordChatExporter:")
        print(f"  After  : {dt_utc.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Before : (leave blank / now)")
        print()
        print(f"CLI flag : --after \"{dt_utc.strftime('%Y-%m-%d %H:%M:%S')}\"")
        if not args:
            print("\nTo ingest a new export, run:")
            print("  python update.py raw/new_messages.json")
        sys.exit(0)

    new_file = args[0]

    if not Path(new_file).exists():
        print(f"Error: File not found: {new_file}")
        sys.exit(1)

    updater = RAGUpdater(store_dir=store_dir)
    updater.update_from_new_file(new_file)
