import torch
import typer
from model import Transformer, Config
from tokenizer import get_tokenizer, TokenizerInterface

from pathlib import Path

default_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device target: ", default_device)


def load_model(checkpoint_path):
    model = Transformer.from_name(checkpoint_path.parent.name)
    print("Found model", model)
    print(model.state_dict())
    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    print("Loaded checkpoint into memory")
    model.load_state_dict(checkpoint, assign=True)
    print("Transferred weights into model")

    model = model.to(device=default_device)
    print("Model -> GPU")
    return model.eval()

def encode_tokens(tokenizer, string, bos=True, device=default_device):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)

def generate(model, tokens):
    pass
app = typer.Typer()

@app.command()
def main(
    prompt: str = typer.Option(...),
    checkpoint_path: Path = typer.Option(...),
):
    model = load_model(checkpoint_path)
    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    assert tokenizer_path.is_file(), str(tokenizer_path)
    tokenizer: TokenizerInterface = get_tokenizer(tokenizer_path, checkpoint_path)

    encoded = encode_tokens(tokenizer, prompt, bos=True, device=default_device)
    print('loaded model and encoded prompt')
    output = model.generate(encoded, max_length=100, temperature=0.7)
    print(tokenizer.decode(output[0], skip_special_tokens=True))

if __name__ == "__main__":
    app()