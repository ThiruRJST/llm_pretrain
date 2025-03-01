import torch

from minimalai.models.gpt import GPT
from minimalai.utils import generate
from minimalai.data import wrapped_tokenizer


def load_model(model_path: str):
    model = GPT(len(wrapped_tokenizer), 512)
    model.load_state_dict(torch.load(model_path))
    model.to("cuda")
    return model


def test_load_model(seed_sentence: str):
    model = load_model("artifacts/best_gpt_step_0.0000.pth")
    assert model is not None
    assert model.eval()
    assert model(seed_sentence) is not None
    print("All tests pass")


if __name__ == "__main__":

    model = load_model("artifacts/best_gpt_step_0.0742.pth")
    model.eval()

    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    seed_tokens = [
        "known't"
    ]
    for seed_token in seed_tokens:
        print(f"Seed token: {seed_token}")
        print(
            generate(
                seed_token,
                wrapped_tokenizer,
                model,
                context_length=256,
                max_new_tokens=10,
            )
        )
