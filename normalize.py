"""
Step 1: Normalize - Parse JSON or email CSV into flat list of messages
"""
import json
import csv
import email
import re
from typing import List, Dict
from pathlib import Path


def _parse_email_message(raw: str, file_path: str = "") -> Dict:
    """Parse a single raw RFC 2822 email into a normalized record."""
    msg = email.message_from_string(raw)

    # Sender: prefer display name, fall back to address.
    from_raw = msg.get("From", "unknown")
    sender_match = re.match(r'^"?([^"<]+)"?\s*<', from_raw)
    if sender_match:
        sender = sender_match.group(1).strip()
    else:
        # address-only format
        sender = re.sub(r'<.*?>', '', from_raw).strip() or from_raw

    # Timestamp from Date header.
    timestamp = msg.get("Date", "")

    # Subject for context.
    subject = msg.get("Subject", "")

    # Body: extract plaintext part.
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                try:
                    body = part.get_payload(decode=True).decode(
                        part.get_content_charset() or "utf-8", errors="replace"
                    )
                    break
                except Exception:
                    body = str(part.get_payload())
                    break
    else:
        try:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode(
                    msg.get_content_charset() or "utf-8", errors="replace"
                )
            else:
                body = str(msg.get_payload())
        except Exception:
            body = str(msg.get_payload())

    body = body.strip()

    # Prepend subject so it's searchable.
    text = f"[{subject}]\n{body}" if subject else body
    text = text.strip()

    return {
        "sender": sender or "unknown",
        "text": text,
        "timestamp": timestamp,
        "subject": subject,
        "file": file_path,
    }


def normalize_email_csv(input_path: str, output_path: str, limit: int = 0) -> List[Dict]:
    """
    Parse a CSV where each row has columns: file, message (raw RFC 2822 email).
    Returns normalized list of messages.

    If limit > 0, stop after collecting that many valid records (reads from the
    beginning of the file — fast early exit instead of reading all 200k rows).
    """
    # Enron emails can have very large fields; raise the limit well above default 131072.
    csv.field_size_limit(10 * 1024 * 1024)  # 10 MB per field
    cap = limit if limit and limit > 0 else None
    print(f"Reading email CSV: {input_path}" + (f" (limit: {cap})" if cap else "") + "...")
    normalized = []
    idx = 0
    with open(input_path, newline='', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if cap and len(normalized) >= cap:
                break
            raw = row.get("message", "")
            file_path = row.get("file", "")
            if not raw.strip():
                continue
            try:
                record = _parse_email_message(raw, file_path)
            except Exception:
                continue
            if not record["text"]:
                continue
            record["id"] = idx
            normalized.append(record)
            idx += 1
            if idx % 1000 == 0:
                print(f"  processed {idx} emails…")

    print(f"Normalized {len(normalized)} emails")
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)
    print(f"Saved to {output_path}")
    return normalized


def normalize_messages(input_path: str, output_path: str, limit: int = 0) -> List[Dict]:
    """
    Parse input file and extract normalized messages.
    Supports: JSON (chat exports), CSV (email datasets).

    Keeps only: sender, message text, timestamp
    One message per record.
    """
    path = Path(input_path)

    # Route to email CSV parser if file ends in .csv
    if path.suffix.lower() == ".csv":
        return normalize_email_csv(input_path, output_path, limit=limit)

    print(f"Reading {input_path}...")

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    normalized = []
    
    # Handle different JSON structures - adapt based on actual structure
    # This is a flexible parser that works with common chat export formats
    
    if isinstance(data, dict):
        if 'messages' in data:
            messages = data['messages']
        elif 'chats' in data:
            messages = []
            for chat in data['chats']:
                if isinstance(chat, dict) and 'messages' in chat:
                    messages.extend(chat['messages'])
        else:
            messages = [data]
    elif isinstance(data, list):
        messages = data
    else:
        raise ValueError("Unknown JSON structure")

    # For JSON, take the LAST limit entries (most recent) before iterating so
    # we never process more rows than needed.
    if limit and limit > 0 and len(messages) > limit:
        messages = messages[-limit:]

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
