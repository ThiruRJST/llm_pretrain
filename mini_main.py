import mlflow.pytorch
import torch
import torch.nn as nn

from minimalai.models.gpt import GPT
from minimalai.utils import generate
from minimalai.data import get_batch, enc
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import autocast, GradScaler  # Add these imports



batch_size = 4
lr = 1e-3
num_epochs = 3
context = 1024
num_steps = 1000

# model
loss_fn = nn.CrossEntropyLoss()
model = GPT(num_tokens=50304, dim=512, num_layers=6, context_length=context).to("cuda")

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2)


scaler = GradScaler()


with mlflow.start_run():
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("lr", lr)
    mlflow.log_param("num_steps", num_steps)
    mlflow.log_param("context_length", context)

    best_loss = float("inf")
    for epoch in range(2):
        step_loss = 0.0
        step_count = 0

        for steps in tqdm(range(num_steps), leave=False):
            xtokens, ytokens = get_batch("train", block_size=context, batch_size=batch_size)

            xtokens = xtokens.to("cuda").long()
            ytokens = ytokens.to("cuda").long()

            optimizer.zero_grad()
            with autocast():
                logits = model(xtokens)
                loss = loss_fn(logits.view(-1, logits.size(-1)), ytokens.view(-1))
            scaler.scale(loss).backward()

            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Step optimizer and update scaler
            scaler.step(optimizer)
            scaler.update()
            
            step_loss += loss.item()
            step_count += 1

            if step_count % 100 == 0:

                loss_value = step_loss / step_count
                if loss_value < best_loss:
                    best_loss = loss_value
                    torch.save(
                        model.state_dict(),
                        f"artifacts/best_gpt_step_{loss_value:.4f}.pth",
                    )
                    print(
                        f"Epoch {epoch}, Step {step_count}, Loss {step_loss/step_count:.3f}"
                    )
                    mlflow.log_metric("loss", step_loss / step_count, step_count)
                    mlflow.pytorch.log_model(
                        model, f"artifacts/best_gpt_step_{loss_value:.4f}.pth"
                    )

                model.eval()

                gen_text = generate(
                    seed_token="Love",
                    model=model,
                    context_length=context,
                    max_new_tokens=100,
                )
                f = open(f"gen_texts/Generated_text_{step_count}.txt", "w")
                f.write(gen_text)

                model.train()
                step_loss = 0.0
                step_count = 0
