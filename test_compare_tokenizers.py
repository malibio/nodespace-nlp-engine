#!/usr/bin/env python3

from tokenizers import Tokenizer

# Load your current tokenizer
current_tokenizer = Tokenizer.from_file("/Users/malibio/nodespace/models/gemma-3-1b-it-onnx/tokenizer.json")

# Load the official Gemma tokenizer
official_tokenizer = Tokenizer.from_file("/Users/malibio/nodespace/models/gemma-1.1-2b-it-test/tokenizer.json")

print("=== TOKENIZER COMPARISON ===")
print(f"Current tokenizer vocab size: {len(current_tokenizer.get_vocab())}")
print(f"Official tokenizer vocab size: {len(official_tokenizer.get_vocab())}")

# Test the problematic token
test_token_id = 2011
current_decode = current_tokenizer.decode([test_token_id])
official_decode = official_tokenizer.decode([test_token_id])

print(f"\nToken 2011 decoding:")
print(f"  Current: '{current_decode}'")
print(f"  Official: '{official_decode}'")

# Test a simple phrase
test_phrase = "Based on the provided context, the income range segment we are targeting is 75,000-150,000 annually."
current_encoded = current_tokenizer.encode(test_phrase)
official_encoded = official_tokenizer.encode(test_phrase)

print(f"\nTest phrase encoding:")
print(f"  Current tokens: {len(current_encoded.ids)} tokens")
print(f"  Official tokens: {len(official_encoded.ids)} tokens")

current_roundtrip = current_tokenizer.decode(current_encoded.ids)
official_roundtrip = official_tokenizer.decode(official_encoded.ids)

print(f"\nRound-trip test:")
print(f"  Current: '{current_roundtrip}'")
print(f"  Official: '{official_roundtrip}'")
print(f"  Match: {current_roundtrip == official_roundtrip}")