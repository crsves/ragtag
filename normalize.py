"""
Step 1: Normalize - Parse JSON into flat list of messages
"""
import json
from typing import List, Dict
from pathlib import Path


def normalize_messages(input_path: str, output_path: str) -> List[Dict]:
    """
    Parse JSON and extract normalized messages.
    
    Keeps only: sender, message text, timestamp
    One message per record.
    """
    print(f"Reading {input_path}...")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    normalized = []
    
    # Handle different JSON structures - adapt based on actual structure
    # This is a flexible parser that works with common chat export formats
    
    if isinstance(data, dict):
        # WhatsApp, Telegram, or similar format
        if 'messages' in data:
            messages = data['messages']
        elif 'chats' in data:
            messages = []
            for chat in data['chats']:
                if isinstance(chat, dict) and 'messages' in chat:
                    messages.extend(chat['messages'])
        else:
            # Try to find message-like structures
            messages = [data]
    elif isinstance(data, list):
        messages = data
    else:
        raise ValueError("Unknown JSON structure")
    
    print(f"Processing {len(messages)} messages...")
    
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
            
        # Extract sender (try different field names)
        sender = (msg.get('sender') or
                 msg.get('from') or
                 msg.get('author') or
                 msg.get('name') or
                 msg.get('from_name') or
                 'unknown')

        # If sender is a dict (e.g. Discord's author object), extract the display name
        if isinstance(sender, dict):
            sender = (sender.get('nickname') or
                     sender.get('name') or
                     sender.get('username') or
                     'unknown')

        # Extract message text (try different field names)
        text = (msg.get('content') or
               msg.get('message') or
               msg.get('text') or
               msg.get('body') or '')
        
        # Handle nested text structures
        if isinstance(text, dict):
            text = text.get('text', str(text))
        elif isinstance(text, list):
            # Some formats store text as array of strings/objects
            text = ' '.join([t if isinstance(t, str) else t.get('text', '') for t in text])
        
        text = str(text).strip()
        
        # Skip empty messages
        if not text:
            continue
        
        # Extract timestamp (try different field names)
        timestamp = (msg.get('timestamp') or 
                    msg.get('date') or 
                    msg.get('time') or 
                    msg.get('created_at') or 
                    msg.get('date_unixtime') or
                    '')
        
        normalized.append({
            'id': idx,
            'sender': str(sender).strip() or 'unknown',
            'text': text,
            'timestamp': str(timestamp)
        })
    
    print(f"Normalized {len(normalized)} messages with text")
    
    # Save normalized data
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)
    
    print(f"Saved to {output_path}")
    return normalized


if __name__ == '__main__':
    INPUT = 'raw/sania.min.json'
    OUTPUT = 'processed/normalized.json'
    
    messages = normalize_messages(INPUT, OUTPUT)
    print(f"\nSample message:")
    if messages:
        print(json.dumps(messages[0], indent=2, ensure_ascii=False))
