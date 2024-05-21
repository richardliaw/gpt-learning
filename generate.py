import torch
import typer
from model import Transformer, Config
from tokenizer import get_tokenizer, TokenizerInterface

from pathlib import Path

default_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device target: ", default_device)


def load_model(checkpoint_path):
    model = Transformer.from_name(checkpoint_path.parent.name)
    precision = torch.bfloat16
    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    print("Loaded checkpoint into memory")
    model.load_state_dict(checkpoint, assign=True)
    print("Transferred weights into model")
    model = model.to(device=default_device, dtype=precision)
    with torch.device(default_device):
        model.setup_caches()
    print("Model -> GPU", model)
    return model.eval()

def encode_tokens(tokenizer, string, bos=True, device=default_device):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)

def logits_to_probs(logits, temp): # [B, 1, Vocab] -> [B, 1, Vocab]
    logits = logits / max(temp, 1e-8)
    return torch.nn.functional.softmax(logits, dim=-1)

def multinomial_sample_one_no_sync(probs_sort): # [B, 1, Vocab] 
    # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def sample(logits, temp=1.0): # -> idx of vocab
    B, T, V = logits.shape
    probs = logits_to_probs(logits[:, -1, :], temp)
    idx = multinomial_sample_one_no_sync(probs)
    assert idx.shape == (B, 1), idx.shape
    return idx, probs


def generate(model, tokened_input, max_length=10):
    tokened_input = tokened_input.to(default_device)
    model = model.to(default_device)
    if len(tokened_input.shape) < 2:
        tokened_input = tokened_input.unsqueeze(0)
    print("Input shape", tokened_input.shape)
    B, T = tokened_input.shape

    x = tokened_input
    temp = torch.ones(B, T, device=default_device)
    input_pos = torch.arange(0, T, device=default_device) * temp
    input_pos = input_pos.to(torch.int32)
    print("input_pos dtype", input_pos.dtype)
    for _ in range(max_length):
        output = model(x, input_pos) # [B, T], [B, T]
        idx = sample(output)[0]
        x = torch.cat([x, idx], dim=-1)
        B, T = x.shape
        temp = torch.ones(B, T, device=default_device)
        input_pos = torch.arange(0, T, device=default_device) * temp
        input_pos = input_pos.to(torch.int32)
    print("Output shape (with prompt)", x.shape)
    return x

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
    output = generate(model, encoded)#, max_length=100, temperature=0.7)
    print(tokenizer.decode(output.tolist()))
    # print(tokenizer.decode(output[0], skip_special_tokens=True))

if __name__ == "__main__":
    app()