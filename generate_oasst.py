"""
Interactive generation for an OASST1-fine-tuned GPT-2 checkpoint.

Examples:
    python generate_oasst.py --checkpoint models/model_best.pt
    python generate_oasst.py --checkpoint models/model_best.pt --prompt "Explain quantum entanglement simply."
    python generate_oasst.py --checkpoint models/model_best.pt --temperature 0.7 --top-k 40

Interactive mode accepts one prompt per line. Type 'exit' or 'quit' to stop.
"""

import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx):
        B, T = idx.shape
        if T > self.config.block_size:
            idx = idx[:, -self.config.block_size:]
            T = idx.shape[1]
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        return self.lm_head(self.transformer.ln_f(x))


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(
        f"Loaded {path} | {config.n_layer}L/{config.n_head}H/"
        f"{config.n_embd}D | step={ckpt.get('step', '?')} | "
        f"val_loss={ckpt.get('val_loss', float('nan')):.4f}"
    )
    return model, config


@torch.no_grad()
def generate(model, enc, prompt, device, max_new_tokens, temperature, top_k, seed):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    formatted = "<|user|>\n" + prompt.strip() + "\n\n<|assistant|>\n"
    ids = enc.encode(formatted)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    original_len = x.shape[1]
    eot = enc.eot_token

    for _ in range(max_new_tokens):
        idx = x[:, -model.config.block_size:]
        logits = model(idx)[:, -1, :]
        logits = logits / max(temperature, 1e-5)

        if top_k is not None and top_k > 0:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < values[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, next_token), dim=1)

        if next_token.item() == eot:
            break

    generated = x[0, original_len:].tolist()
    text = enc.decode(generated)

    # OASST responses end at EOS. Also stop if the model starts a new user turn.
    for marker in ("<|user|>", "\n<|user|>"):
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="log/oasst/model_best.pt")
    p.add_argument("--prompt", default=None)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    enc = tiktoken.get_encoding("gpt2")
    model, _ = load_model(args.checkpoint, device)

    if args.prompt is not None:
        print("\nAssistant:\n")
        print(generate(model, enc, args.prompt, device, args.max_new_tokens, args.temperature, args.top_k, args.seed))
        return

    print("\nInteractive OASST mode. Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.lower() in {"exit", "quit"}:
            break
        if not prompt:
            continue
        answer = generate(model, enc, prompt, device, args.max_new_tokens, args.temperature, args.top_k, args.seed)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()
