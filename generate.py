import torch
import typer
from model import Transformer, Config

from pathlib import Path



def load_model(checkpoint_path):
    model = Transformer.from_name(checkpoint_path.parent.name)
    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    model.load_state_dict(checkpoint, assign=True)
    model = model.to(device=device)
    return model.eval()


app = typer.Typer()

@app.command()
def generate(
    prompt: str = typer.Option(...),
    checkpoint_path: Path = typer.Option(...),
):
    model = load_model(checkpoint_path)
    encoded = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(encoded, max_length=100, temperature=0.7)
    print(tokenizer.decode(output[0], skip_special_tokens=True))

if __name__ == "__main__":
    app()