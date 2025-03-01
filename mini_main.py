import mlflow.pytorch
import torch
import torch.nn as nn

from minimalai.models.gpt import GPT
from minimalai.utils import generate
from minimalai.data import create_batches, wrapped_tokenizer, data
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

batch_size = 32
lr = 1e-3
num_epochs = 3
context = 256
num_steps = 1000

loss_fn = nn.CrossEntropyLoss()
model = GPT(len(wrapped_tokenizer), 512).to("cuda")

torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2)

with mlflow.start_run():
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("lr", lr)
    mlflow.log_param("num_steps", num_steps)
    mlflow.log_param("context_length", context)


    for epoch in range(2):
        step_loss = 0.0
        step_count = 0

        for steps in tqdm(range(num_steps), leave=False):
            xtokens, ytokens = create_batches(data, batch_size, context)
            
            xtokens = xtokens.to("cuda").long()
            ytokens = ytokens.to("cuda").long()

            optimizer.zero_grad()
            logits = model(xtokens)
            loss = loss_fn(logits.view(-1, logits.size(-1)), ytokens.view(-1))
            loss.backward()
            optimizer.step()
            step_loss += loss.item()
            step_count += 1

            if step_count % 100 == 0:
                model.eval()
                print(f"Epoch {epoch}, Step {step_count}, Loss {step_loss/step_count:.3f}")
                mlflow.log_metric("loss", step_loss/step_count, step_count)
                gen_text = generate(
                    seed_token="Love",
                    tokenizer=wrapped_tokenizer,
                    model=model,
                    context_length=context,
                    max_new_tokens=100,
                )
                print(gen_text)
                model.train()
                step_loss = 0.0