"""
Step 2: Chunk messages - Create self-contained chunks
"""
import json
from datetime import datetime
from typing import List, Dict
from pathlib import Path


def _fmt_ts(ts: str) -> str:
    """Format ISO timestamp to 'YYYY-MM-DD HH:MM'."""
    try:
        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M')
    except Exception:
        return ts[:16] if ts else ''


def chunk_messages(normalized_messages: List[Dict], 
                   messages_per_chunk: int = 1,
                   overlap: int = 0) -> List[Dict]:
    """
    Create chunks from normalized messages.
    
    Args:
        normalized_messages: List of normalized message dicts
        messages_per_chunk: How many messages to bundle (1 = one chunk per message)
        overlap: How many messages to overlap between chunks (for context)
    
    Returns:
        List of chunk dicts with metadata
    """
    chunks = []
    
    for i in range(0, len(normalized_messages), messages_per_chunk - overlap):
        # Get messages for this chunk
        chunk_messages = normalized_messages[i:i + messages_per_chunk]
        
        if not chunk_messages:
            break
        
        # Combine text from all messages in chunk.
        # Format: "[YYYY-MM-DD HH:MM] sender: message"
        # This is what gets embedded — clean, no metadata noise.
        combined_text = '\n'.join([
            f"[{_fmt_ts(msg['timestamp'])}] {msg['sender']}: {msg['text']}"
            for msg in chunk_messages
        ])
        
        # Get timestamp range
        timestamps = [msg['timestamp'] for msg in chunk_messages if msg['timestamp']]
        timestamp_start = timestamps[0] if timestamps else ''
        timestamp_end = timestamps[-1] if timestamps else ''
        
        # Get unique senders
        senders = list(set([msg['sender'] for msg in chunk_messages]))
        
        chunk = {
            'chunk_id': len(chunks),
            'text': combined_text,
            'sender': senders[0] if len(senders) == 1 else ', '.join(senders),
            'senders': senders,
            'timestamp_start': timestamp_start,
            'timestamp_end': timestamp_end,
            'message_ids': [msg['id'] for msg in chunk_messages],
            'message_count': len(chunk_messages)
        }
        
        chunks.append(chunk)
        
        # Break if we've processed all messages
        if i + messages_per_chunk >= len(normalized_messages):
            break
    
    return chunks


def save_chunks(chunks: List[Dict], output_path: str):
    """Save chunks to JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(chunks)} chunks to {output_path}")


if __name__ == '__main__':
    # Load normalized messages
    with open('processed/normalized.json', 'r', encoding='utf-8') as f:
        messages = json.load(f)
    
    print(f"Loaded {len(messages)} normalized messages")
    
    # Create chunks (1 message per chunk for simplicity)
    chunks = chunk_messages(messages, messages_per_chunk=1)
    
    save_chunks(chunks, 'processed/chunks.json')
    
    print(f"\nSample chunk:")
    if chunks:
        print(json.dumps(chunks[0], indent=2, ensure_ascii=False))
