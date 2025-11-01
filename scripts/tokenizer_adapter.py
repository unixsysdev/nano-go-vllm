#!/usr/bin/env python3
import sys, json
from pathlib import Path

try:
    from tokenizers import Tokenizer
except Exception as e:
    print(json.dumps({"error": f"python tokenizers missing: {e}"}))
    sys.exit(1)

def load_tok(model_dir: str):
    tok_file = Path(model_dir) / 'tokenizer.json'
    if not tok_file.exists():
        print(json.dumps({"error": f"tokenizer.json not found in {model_dir}"}))
        sys.exit(2)
    return Tokenizer.from_file(str(tok_file))

def main():
    if len(sys.argv) < 4:
        print(json.dumps({"error": "usage: tokenizer_adapter.py <encode|decode> <model_dir> <payload>"}))
        sys.exit(2)
    mode = sys.argv[1]
    model_dir = sys.argv[2]
    payload = sys.argv[3]
    tok = load_tok(model_dir)
    if mode == 'encode':
        enc = tok.encode(payload)
        print(json.dumps({"ids": enc.ids}))
    elif mode == 'decode':
        try:
            ids = json.loads(payload)
        except Exception:
            ids = [int(x) for x in payload.split(',') if x]
        s = tok.decode(ids)
        print(json.dumps({"text": s}))
    else:
        print(json.dumps({"error": f"unknown mode {mode}"}))
        sys.exit(2)

if __name__ == '__main__':
    main()

