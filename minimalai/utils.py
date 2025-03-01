import torch
import torch.nn.functional as F

def generate(seed_token, tokenizer, model, context_length, max_new_tokens, temperature=1.0):
    model.eval()
    gen_tokens = []
    tokens = tokenizer(seed_token, padding="max_length", truncation=True, max_length=context_length, return_tensors="pt")["input_ids"]
    
    for _ in range(max_new_tokens):
        # Prepare input
        x = tokens[-context_length:].to("cuda:0").long()
        
        # Get predictions
        logits = model(x)
        
        # Sample next token (from the last position)
        logits = logits[0, -1, :] / temperature
        probs = F.softmax(logits, dim=0)
        next_token = torch.multinomial(probs, num_samples=1).item()
        
        gen_tokens.append(next_token)
    
    # Convert to text
    return ''.join([tokenizer.decode([t]) for t in gen_tokens])