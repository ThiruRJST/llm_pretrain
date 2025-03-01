import torch
import torch.nn.functional as F
import tiktoken

enc = tiktoken.get_encoding("gpt2")

def generate(
    seed_token, model, context_length, max_new_tokens, temperature=1.0
):
    model.eval()
    gen_tokens = []
    gen_tokens.extend(enc.encode_ordinary(seed_token))

    for i in range(max_new_tokens):
        # Prepare input
        input_tokens = torch.tensor(gen_tokens[-context_length:]).unsqueeze(0)
        print(input_tokens.shape)
        x = input_tokens.to("cuda:0").long()

        # Get predictions
        logits = model(x)

        # Sample next token (from the last position)
        logits = logits[0, -1, :] / temperature
        probs = F.softmax(logits, dim=0)
        next_token = torch.multinomial(probs, num_samples=1).item()

        gen_tokens.append(next_token)


    return  "".join([(enc.decode([t])) for t in gen_tokens])
