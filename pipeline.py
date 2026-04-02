"""
Complete RAG Pipeline - Run all steps in sequence
"""
import json
from pathlib import Path
from normalize import normalize_messages
from chunk import chunk_messages, save_chunks
from embed import EmbeddingGenerator, save_embeddings
from store import VectorStore


def build_rag_system(
    input_file: str = 'raw/sania.min.json',
    messages_per_chunk: int = 1,
    model_name: str = 'all-MiniLM-L6-v2',
    output_dir: str = 'processed',
):
    """
    Build complete RAG system from raw JSON.

    Args:
        input_file: Path to raw messages JSON
        messages_per_chunk: Chunking strategy (1 = one message per chunk)
        model_name: Embedding model to use
        output_dir: Root directory for all processed output (default: 'processed')
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("BUILDING RAG SYSTEM")
    print("="*80)

    # Step 1: Normalize
    print("\n[STEP 1/5] Normalizing messages...")
    normalized = normalize_messages(input_file, str(out / 'normalized.json'))
    print(f"✓ Normalized {len(normalized)} messages")

    # Step 2: Chunk
    print("\n[STEP 2/5] Chunking messages...")
    chunks = chunk_messages(normalized, messages_per_chunk=messages_per_chunk)
    save_chunks(chunks, str(out / 'chunks.json'))
    print(f"✓ Created {len(chunks)} chunks")

    # Step 3: Generate embeddings
    print("\n[STEP 3/5] Generating embeddings...")
    generator = EmbeddingGenerator(model_name)
    embeddings = generator.embed_chunks(chunks)
    save_embeddings(embeddings, chunks, str(out / 'embeddings'))
    print(f"✓ Generated embeddings with shape {embeddings.shape}")

    # Step 4: Build vector store
    print("\n[STEP 4/5] Building vector store...")
    store = VectorStore(embedding_dim=embeddings.shape[1])
    store.add(embeddings, chunks)
    store.save(str(out / 'vector_store'))
    print(f"✓ Built vector store with {store.index.ntotal} vectors")

    # Step 5: Summary
    print("\n[STEP 5/5] Summary")
    print("="*80)
    print(f"✓ Input messages: {len(normalized)}")
    print(f"✓ Chunks created: {len(chunks)}")
    print(f"✓ Embedding dimension: {embeddings.shape[1]}")
    print(f"✓ Model used: {model_name}")
    print(f"✓ Storage location: {out / 'vector_store'}")
    print("="*80)

    print("\n✓ RAG system built successfully!")
    print()


if __name__ == '__main__':
    import os as _os_env
    # Suppress multiprocessing resource_tracker's "leaked semaphore" warning that
    # appears in frozen binaries when os._exit() terminates worker processes
    # before they can release their semaphores. The OS reclaims the resources
    # automatically; the warning is purely cosmetic.
    _os_env.environ.setdefault(
        'PYTHONWARNINGS',
        'ignore::UserWarning:multiprocessing.resource_tracker',
    )

    import sys
    import glob
    import multiprocessing

    # Required by PyInstaller so that frozen-exe worker processes spawned by
    # multiprocessing (e.g. from HuggingFace model downloads) are handled
    # correctly rather than re-running main().
    multiprocessing.freeze_support()

    input_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    if not input_file:
        # Try to find a json file in raw/
        raw_files = glob.glob('raw/*.json')
        if raw_files:
            input_file = raw_files[0]
        else:
            print("Error: No input file specified and no JSON file found in raw/")
            print(f"Usage: python pipeline.py [input_file] [output_dir]")
            sys.exit(1)

    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'processed'

    if not Path(input_file).exists():
        print(f"Error: File not found: {input_file}")
        print(f"\nUsage: python pipeline.py [input_file] [output_dir]")
        sys.exit(1)

    build_rag_system(input_file, output_dir=output_dir)

    # When running as a PyInstaller frozen binary, Python's module-cleanup
    # phase triggers re-imports of lazy-loaded transformers submodules whose
    # native prerequisites may have already been finalised, causing noisy
    # tracebacks even though the pipeline succeeded. os._exit() exits
    # immediately after work is done, bypassing cleanup. Flush first so
    # all print output is visible before the process terminates.
    import sys as _sys
    if getattr(_sys, 'frozen', False):
        _sys.stdout.flush()
        _sys.stderr.flush()
        # Kill the multiprocessing resource_tracker background process before
        # exiting. It otherwise detects "leaked" semaphores (created by
        # torch internals) and prints a UserWarning after we exit. The OS
        # reclaims those semaphores automatically when the process group dies.
        try:
            import multiprocessing.resource_tracker as _mrt
            _rt = getattr(_mrt, '_resource_tracker', None)
            if _rt is not None:
                _pid = getattr(_rt, '_pid', None)
                if _pid:
                    import os as _os2, signal as _sig
                    _os2.kill(_pid, _sig.SIGKILL)
        except Exception:
            pass
        import os as _os
        _os._exit(0)
