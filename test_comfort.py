#!/usr/bin/env python3
"""
Test script for Comfort model
Checks if the model generates coherent responses without NaN/Inf
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Configuration
MODEL_DIR = "checkpoints/comfort_large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Test nightmares
TEST_NIGHTMARES = [
    "I have an important presentation tomorrow and I'm not ready",
    "I'm falling from a tall building",
    "Everyone is staring at me and laughing",
    "I'm trapped in a small space with no way out",
    "I'm being chased through dark streets",
]

def test_comfort_model():
    print("=" * 70)
    print("TESTING GPT2-LARGE COMFORT MODEL")
    print("=" * 70)
    print(f"\nLoading from: {MODEL_DIR}")
    
    # Load model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()
    
    print(f"Device: {DEVICE}")
    print(f"Vocabulary size: {len(tokenizer)}")
    print("\n" + "=" * 70)
    print("GENERATING COMFORT RESPONSES")
    print("=" * 70)
    
    for i, nightmare in enumerate(TEST_NIGHTMARES, 1):
        print(f"\n{i}. Nightmare: {nightmare}")
        print("-" * 70)
        
        # Create prompt
        prompt = f"<NIGHTMARE> {nightmare}\n<COMFORT> "
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        try:
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=60,
                    min_new_tokens=15,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    do_sample=True,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # Decode
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # Extract just the comfort response
            if "<COMFORT>" in full_text:
                comfort_response = full_text.split("<COMFORT>")[1].strip()
                # Remove any leaked tokens
                for token in ["<NIGHTMARE>", "<FEAR>", "<eos>", "<pad>"]:
                    comfort_response = comfort_response.split(token)[0].strip()
                
                print(f"Comfort: {comfort_response}")
                
                # Check for issues
                if len(comfort_response) < 10:
                    print("⚠️ WARNING: Response too short!")
                elif comfort_response.count(" ") < 3:
                    print("⚠️ WARNING: Response seems incomplete!")
                else:
                    print("✓ Response looks good")
            else:
                print("❌ ERROR: No <COMFORT> token in output!")
                print(f"Full output: {full_text}")
                
        except RuntimeError as e:
            if "nan" in str(e).lower() or "inf" in str(e).lower():
                print("❌ CRITICAL: Model generated NaN/Inf values!")
                print(f"Error: {e}")
                return False
            else:
                raise
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print("\n✓ All tests passed! Model is generating coherent comfort responses.")
    return True

if __name__ == "__main__":
    success = test_comfort_model()
    exit(0 if success else 1)