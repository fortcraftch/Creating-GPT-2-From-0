"""
Fine-tune a GPT-2-style model on the English portion of OpenAssistant/oasst1.

This script is self-contained:
  1. Downloads/caches OpenAssistant/oasst1 through Hugging Face Datasets.
  2. Keeps English messages only.
  3. Reconstructs root -> assistant-response conversation examples.
  4. Tokenizes them with the GPT-2 tokenizer.
  5. Writes fixed-length training/validation shards under data/oasst1_shards/.
  6. Fine-tunes a local GPT checkpoint, computing loss ONLY on assistant tokens.

The generated checkpoint is compatible with the project's generate.py/eval.py:
    {
        "model": state_dict,
        "config": {"block_size", "vocab_size", "n_layer", "n_head", "n_embd"},
        "step": ...,
        "val_loss": ...,
    }

Recommended first run:
    python finetune_oasst.py --checkpoint gpt2Base.pt --steps 2000

Resume:
    python finetune_oasst.py --checkpoint log/oasst/model_2000.pt --resume

Rebuild only the dataset:
    python finetune_oasst.py --build-data-only

Dependencies:
    pip install torch tiktoken datasets tqdm numpy
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


# -----------------------------------------------------------------------------
# GPT-2 model (same architecture as the user's base GPT-2 training code)
# -----------------------------------------------------------------------------

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
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

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

    def forward(self, idx, targets=None, loss_mask=None):
        B, T = idx.size()
        if T > self.config.block_size:
            raise ValueError("sequence exceeds block_size")
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        logits = self.lm_head(self.transformer.ln_f(x))
        loss = None
        if targets is not None:
            token_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="none",
            ).view(B, T)
            if loss_mask is None:
                loss = token_loss.mean()
            else:
                loss = (token_loss * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)
        return logits, loss


# -----------------------------------------------------------------------------
# OASST1 -> fixed-length shards
# -----------------------------------------------------------------------------

SPECIAL_USER = "<|user|>\n"
SPECIAL_ASSISTANT = "<|assistant|>\n"


def load_oasst_split(split):
    """Load the current Parquet-backed OASST1 split."""
    # OASST1 is currently Parquet-backed on HF and exposes train/validation.
    return load_dataset("OpenAssistant/oasst1", split=split)


def build_examples(dataset, max_examples=None):
    """Reconstruct English root-to-assistant-turn examples from flat messages.

    For every assistant message, its complete ancestor chain is used as context.
    This gives the model both single-turn and multi-turn SFT examples without
    needing the separate .trees.jsonl.gz files.
    """
    rows = {}
    children = {}

    for row in dataset:
        if row.get("lang") != "en":
            continue
        if row.get("deleted", False):
            continue
        mid = row["message_id"]
        rows[mid] = row
        parent = row.get("parent_id")
        if parent:
            children.setdefault(parent, []).append(mid)

    # Cache reconstructed chains.
    chain_cache = {}

    def chain(mid):
        if mid in chain_cache:
            return chain_cache[mid]
        row = rows[mid]
        parent = row.get("parent_id")
        if parent and parent in rows:
            result = chain(parent) + [mid]
        else:
            result = [mid]
        chain_cache[mid] = result
        return result

    examples = []
    for mid, row in rows.items():
        if row.get("role") != "assistant":
            continue
        ids = chain(mid)
        messages = [rows[x] for x in ids]
        # A valid SFT example should start with a prompter and alternate roles.
        if not messages or messages[0].get("role") != "prompter":
            continue
        valid = True
        for i, msg in enumerate(messages):
            expected = "prompter" if i % 2 == 0 else "assistant"
            if msg.get("role") != expected:
                valid = False
                break
            if not isinstance(msg.get("text"), str) or not msg["text"].strip():
                valid = False
                break
        if not valid:
            continue

        examples.append(messages)
        if max_examples and len(examples) >= max_examples:
            break

    return examples


def encode_conversation(messages, enc, block_size):
    """Encode one conversation ending in an assistant message.

    Returns fixed-size token and mask arrays. Prompt tokens have mask=0;
    assistant tokens + EOS have mask=1.
    """
    token_ids = []
    loss_mask = []
    eot = enc.eot_token

    for msg in messages:
        if msg["role"] == "prompter":
            prefix = enc.encode(SPECIAL_USER)
            text_tokens = enc.encode(msg["text"].strip() + "\n")
            token_ids.extend(prefix + text_tokens)
            loss_mask.extend([0] * (len(prefix) + len(text_tokens)))
        else:
            prefix = enc.encode(SPECIAL_ASSISTANT)
            text_tokens = enc.encode(msg["text"].strip())
            part = prefix + text_tokens + [eot]
            token_ids.extend(part)
            loss_mask.extend([0] * len(prefix) + [1] * (len(text_tokens) + 1))

    # Keep the target answer. If the complete conversation is too long, trim
    # from the left; this preserves the final assistant response as much as
    # possible, which is what the loss is training on.
    if len(token_ids) > block_size:
        token_ids = token_ids[-block_size:]
        loss_mask = loss_mask[-block_size:]

    # We need a next-token target, so pad to block_size+1.
    if len(token_ids) < 2:
        return None

    # For fixed-shape training, create idx of block_size and targets of block_size.
    # If shorter than block_size, pad with EOS and mask=0.
    idx = token_ids[:-1]
    targets = token_ids[1:]
    mask = loss_mask[1:]

    pad_len = block_size - len(idx)
    if pad_len > 0:
        idx += [eot] * pad_len
        targets += [eot] * pad_len
        mask += [0] * pad_len

    return (
        np.asarray(idx, dtype=np.uint16),
        np.asarray(targets, dtype=np.uint16),
        np.asarray(mask, dtype=np.uint8),
    )


def write_shards(examples, enc, block_size, out_dir, prefix, shard_size):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale shards of this split so rebuilding cannot silently mix data.
    for p in out_dir.glob(f"{prefix}-*.npz"):
        p.unlink()

    shard_idx = 0
    count = 0
    buffers = [[], [], []]

    def flush():
        nonlocal shard_idx, count
        if not buffers[0]:
            return
        tokens = np.stack(buffers[0])
        targets = np.stack(buffers[1])
        masks = np.stack(buffers[2])
        path = out_dir / f"{prefix}-{shard_idx:05d}.npz"
        np.savez(path, tokens=tokens, targets=targets, masks=masks)
        count += len(tokens)
        shard_idx += 1
        buffers[0].clear(); buffers[1].clear(); buffers[2].clear()

    for messages in tqdm(examples, desc=f"Tokenizing {prefix}"):
        item = encode_conversation(messages, enc, block_size)
        if item is None:
            continue
        for i, arr in enumerate(item):
            buffers[i].append(arr)
        if len(buffers[0]) >= shard_size:
            flush()

    flush()
    return count, shard_idx


def prepare_data(args, enc):
    out_dir = Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "metadata.json"

    if marker.exists() and not args.rebuild_data:
        meta = json.loads(marker.read_text(encoding="utf-8"))
        if meta.get("block_size") == args.block_size:
            print(f"Using existing OASST shards in {out_dir}")
            return meta

    print("Downloading/loading OpenAssistant/oasst1...")
    train_ds = load_oasst_split("train")
    val_ds = load_oasst_split("validation")

    print(f"Raw train: {len(train_ds):,} messages")
    print(f"Raw validation: {len(val_ds):,} messages")

    train_examples = build_examples(train_ds, args.max_train_examples)
    val_examples = build_examples(val_ds, args.max_val_examples)

    print(f"English reconstructed train examples: {len(train_examples):,}")
    print(f"English reconstructed validation examples: {len(val_examples):,}")

    train_count, train_shards = write_shards(
        train_examples, enc, args.block_size, out_dir, "train", args.shard_size
    )
    val_count, val_shards = write_shards(
        val_examples, enc, args.block_size, out_dir, "val", args.shard_size
    )

    meta = {
        "dataset": "OpenAssistant/oasst1",
        "language": "en",
        "block_size": args.block_size,
        "train_examples": train_count,
        "val_examples": val_count,
        "train_shards": train_shards,
        "val_shards": val_shards,
        "format": "npz(tokens, targets, masks)",
    }
    marker.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return meta


# -----------------------------------------------------------------------------
# Sharded batch loader
# -----------------------------------------------------------------------------

class ShardLoader:
    def __init__(self, data_dir, split, batch_size, device, seed=1337):
        self.data_dir = Path(data_dir)
        self.split = split
        self.batch_size = batch_size
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.shards = sorted(self.data_dir.glob(f"{split}-*.npz"))
        if not self.shards:
            raise FileNotFoundError(f"No {split} shards in {self.data_dir}")
        self.arrays = [np.load(p, mmap_mode="r") for p in self.shards]
        self.lengths = [len(x["tokens"]) for x in self.arrays]
        self.total = sum(self.lengths)
        self.cumulative = np.cumsum(self.lengths)
        print(f"{split}: {len(self.shards)} shards, {self.total:,} examples")

    def _locate(self, indices):
        shard_ids = np.searchsorted(self.cumulative, indices, side="right")
        prev = np.concatenate(([0], self.cumulative[:-1]))
        local = indices - prev[shard_ids]
        return shard_ids, local

    def get_batch(self):
        indices = self.rng.integers(0, self.total, size=self.batch_size)
        shard_ids, local_ids = self._locate(indices)
        tokens = np.empty((self.batch_size, self.arrays[0]["tokens"].shape[1]), dtype=np.uint16)
        targets = np.empty_like(tokens)
        masks = np.empty((self.batch_size, tokens.shape[1]), dtype=np.uint8)
        for sid in np.unique(shard_ids):
            sel = np.where(shard_ids == sid)[0]
            li = local_ids[sel]
            tokens[sel] = self.arrays[sid]["tokens"][li]
            targets[sel] = self.arrays[sid]["targets"][li]
            masks[sel] = self.arrays[sid]["masks"][li]
        x = torch.from_numpy(tokens.astype(np.int64)).to(self.device)
        y = torch.from_numpy(targets.astype(np.int64)).to(self.device)
        m = torch.from_numpy(masks.astype(np.float32)).to(self.device)
        return x, y, m


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model" not in ckpt or "config" not in ckpt:
        raise ValueError("Checkpoint must contain 'model' and 'config'.")
    config = GPTConfig(**ckpt["config"])
    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    return model, config, ckpt


def get_lr(step, warmup, total_steps, max_lr, min_lr):
    if step < warmup:
        return max_lr * (step + 1) / max(1, warmup)
    if total_steps <= warmup:
        return min_lr
    progress = (step - warmup) / max(1, total_steps - warmup)
    progress = min(max(progress, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


def evaluate(model, loader, steps, autocast_dtype):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(steps):
            x, y, m = loader.get_batch()
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(x.device.type == "cuda")):
                _, loss = model(x, y, m)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(model, config, optimizer, step, val_loss, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": vars(config),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "val_loss": val_loss,
    }, path)


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    enc = tiktoken.get_encoding("gpt2")

    # Build data before loading the model so --build-data-only needs no checkpoint.
    prepare_data(args, enc)
    if args.build_data_only:
        return

    model, config, original_ckpt = load_checkpoint(args.checkpoint, device)
    if config.block_size != args.block_size:
        raise ValueError(
            f"Checkpoint block_size={config.block_size}, but shard block_size={args.block_size}. "
            "Use --block-size matching the checkpoint."
        )

    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    start_step = 0
    if args.resume and original_ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(original_ckpt["optimizer"])
        start_step = int(original_ckpt.get("step", 0))
        print(f"Resuming optimizer at step {start_step}")

    train_loader = ShardLoader(args.data_dir, "train", args.batch_size, device, args.seed)
    val_loader = ShardLoader(args.data_dir, "val", args.batch_size, device, args.seed + 1)

    use_cuda = device.type == "cuda"
    bf16 = use_cuda and torch.cuda.is_bf16_supported()
    autocast_dtype = torch.bfloat16 if bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_cuda and not bf16))

    if args.compile and hasattr(torch, "compile"):
        print("Compiling model...")
        model = torch.compile(model)

    print(f"Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"AMP: {'bf16' if bf16 else ('fp16' if use_cuda else 'off')}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    t0 = time.time()

    for step in range(start_step, args.steps):
        lr = get_lr(step, args.warmup_steps, args.steps, args.lr, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(args.grad_accum):
            x, y, m = train_loader.get_batch()
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_cuda):
                _, loss = model(x, y, m)
                loss = loss / args.grad_accum
            accum_loss += loss.item()
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        real_step = step + 1
        if real_step % args.log_interval == 0 or real_step == 1:
            elapsed = time.time() - t0
            print(
                f"step {real_step:6d}/{args.steps} | "
                f"lr {lr:.2e} | train_loss {accum_loss:.4f} | "
                f"tok/s ~{args.batch_size * args.grad_accum * config.block_size / max(elapsed, 1e-6):.0f}"
            )
            t0 = time.time()

        if real_step % args.eval_interval == 0 or real_step == args.steps:
            val_loss = evaluate(model, val_loader, args.eval_steps, autocast_dtype)
            print(f"validation loss: {val_loss:.4f}")

            # compiled models expose the original module through _orig_mod
            save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            ckpt_path = out_dir / f"model_{real_step}.pt"
            save_checkpoint(save_model, config, optimizer, real_step, val_loss, ckpt_path)
            print(f"saved: {ckpt_path}")
            if val_loss < best_val:
                best_val = val_loss
                best_path = out_dir / "model_best.pt"
                save_checkpoint(save_model, config, optimizer, real_step, val_loss, best_path)
                print(f"saved best: {best_path}")

    print("Fine-tuning complete.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="gpt2Base.pt")
    p.add_argument("--data-dir", default="data/oasst1_shards")
    p.add_argument("--out-dir", default="log/oasst")
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--min-lr", type=float, default=5e-6)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--shard-size", type=int, default=2000)
    p.add_argument("--max-train-examples", type=int, default=None)
    p.add_argument("--max-val-examples", type=int, default=None)
    p.add_argument("--rebuild-data", action="store_true")
    p.add_argument("--build-data-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
