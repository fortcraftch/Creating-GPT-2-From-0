"""
eval.py - Benchmark evaluation for the GPT project.

Supports:
  * local checkpoints produced by train_gpt2.py
  * Hugging Face GPT-2 models: gpt2, gpt2-medium, gpt2-large, gpt2-xl

Core benchmark families used in / closely related to the GPT-3 evaluation:
  HellaSwag, LAMBADA, PIQA, OpenBookQA, ARC-Easy, ARC-Challenge,
  WinoGrande, SuperGLUE BoolQ, SuperGLUE RTE, WSC, TriviaQA,
  StoryCloze, WebQuestions, COPA, RACE, MMLU, GSM8K, TruthfulQA, PTB,
  plus a small synthetic arithmetic evaluation.

Install dependencies in your project environment:
    pip install -U datasets transformers huggingface_hub requests tqdm tiktoken

Examples:
    python eval.py --model log/model_19072.pt
    python eval.py --model log/model_19072.pt --tasks hellaswag,lambada,piqa,mmlu,gsm8k
    python eval.py --model gpt2-xl --tasks all --limit 1000
"""

import argparse
import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from tqdm import tqdm


# =============================================================================
# Model Architecture & Adapter
# =============================================================================

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
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))


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
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "wpe": nn.Embedding(config.block_size, config.n_embd),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": nn.LayerNorm(config.n_embd),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx):
        _, T = idx.size()
        if T > self.config.block_size:
            raise ValueError(f"Input length {T} > block_size {self.config.block_size}")
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        return self.lm_head(self.transformer.ln_f(x))


class Adapter:
    def __init__(self, model, config, device, name, enc):
        self.model = model.eval()
        self.config = config
        self.device = device
        self.name = name
        self.enc = enc

    @property
    def block_size(self):
        return getattr(self.config, "block_size", getattr(self.config, "n_positions", 1024))

    @property
    def params(self):
        return sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def logits(self, ids):
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids, dtype=torch.long, device=self.device)
        else:
            ids = ids.to(self.device)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if ids.size(1) > self.block_size:
            ids = ids[:, -self.block_size:]
        out = self.model(ids)
        if isinstance(out, tuple):
            out = out[0]
        if hasattr(out, "logits"):
            out = out.logits
        return out

    @torch.no_grad()
    def completion_losses(self, ids, completion_start):
        """Return (sum NLL, mean NLL) over the completion tokens only."""
        full_len = len(ids)
        if full_len < 2:
            return float("inf"), float("inf")
        if full_len > self.block_size:
            trim = full_len - self.block_size
            ids = ids[-self.block_size:]
            completion_start = max(1, completion_start - trim)
        x = ids[:-1]
        y = torch.tensor(ids[1:], dtype=torch.long, device=self.device)
        logits = self.logits(x)
        losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y, reduction="none")
        first = max(0, completion_start - 1)
        losses = losses[first:]
        if losses.numel() == 0:
            return float("inf"), float("inf")
        return losses.sum().item(), losses.mean().item()

    @torch.no_grad()
    def greedy_generate(self, prompt, max_new_tokens=20):
        ids = self.enc.encode(prompt)
        out = list(ids)
        for _ in range(max_new_tokens):
            logits = self.logits(out)
            nxt = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            out.append(nxt)
            if nxt == self.enc.eot_token:
                break
        return self.enc.decode(out[len(ids):]).strip()


def load_model(identifier, device, enc):
    p = Path(identifier)
    if p.exists() and p.is_file():
        ckpt = torch.load(p, map_location=device, weights_only=False)
        if "model" not in ckpt or "config" not in ckpt:
            raise ValueError(f"{p} is not a project checkpoint")
        cfg = GPTConfig(**ckpt["config"])
        model = GPT(cfg)
        model.load_state_dict(ckpt["model"])
        model.to(device).eval()
        return Adapter(model, cfg, device, str(p), enc)

    try:
        from transformers import GPT2LMHeadModel
    except ImportError as exc:
        raise RuntimeError("Install transformers: pip install transformers") from exc

    model = GPT2LMHeadModel.from_pretrained(identifier).to(device).eval()
    return Adapter(model, model.config, device, identifier, enc)


# =============================================================================
# Dataset loading / helpers
# =============================================================================

def ds(path, name=None, split="validation"):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install/update datasets: pip install -U datasets"
        ) from exc

    kwargs = {
        "path": path,
        "split": split,
    }

    if name:
        kwargs["name"] = name

    try:
        return load_dataset(**kwargs)
    except Exception as exc:
        config_text = f" / config '{name}'" if name else ""
        raise RuntimeError(
            f"Could not load dataset '{path}'{config_text} "
            f"/ split '{split}'. Error: {exc}"
        ) from exc


def limited(dataset, limit):
    if limit is None:
        return dataset
    return dataset.select(range(min(limit, len(dataset))))


def mc_score(model, context, completion, normalized=True):
    c = model.enc.encode(context)
    a = model.enc.encode(completion)
    if not a:
        return float("inf")
    return model.completion_losses(c + a, len(c))[1 if normalized else 0]


def normalize_answer(s):
    s = str(s).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return " ".join(s.split())


# =============================================================================
# Existing Evaluation Functions
# =============================================================================

def eval_hellaswag(model, limit):
    data = limited(ds("allenai/hellaswag", split="validation"), limit)
    correct_sum = correct_norm = total = 0
    for ex in tqdm(data, desc="HellaSwag"):
        losses_sum, losses_norm = [], []
        for ending in ex["endings"]:
            c = model.enc.encode(ex["ctx"])
            a = model.enc.encode(" " + ending)
            s, m = model.completion_losses(c + a, len(c))
            losses_sum.append(s)
            losses_norm.append(m)
        pred_s = min(range(4), key=losses_sum.__getitem__)
        pred_n = min(range(4), key=losses_norm.__getitem__)
        label = int(ex["label"])
        correct_sum += pred_s == label
        correct_norm += pred_n == label
        total += 1
    return {"accuracy": correct_sum / total, "accuracy_norm": correct_norm / total, "total": total}


def eval_lambada(model, limit):
    data = ds("EleutherAI/lambada_openai", "default", "test")
    data = limited(data, limit)
    correct = total = target_nll = target_tokens = 0
    for ex in tqdm(data, desc="LAMBADA"):
        text = ex.get("text") or ex.get("sentence") or ex.get("context")
        if not text:
            continue
        words = text.rstrip().split()
        if len(words) < 2:
            continue
        context, target = " ".join(words[:-1]), words[-1]
        c = model.enc.encode(context)
        a = model.enc.encode(" " + target)
        s, _ = model.completion_losses(c + a, len(c))
        generated = list(c)
        pred = []
        for _ in a:
            logits = model.logits(generated)
            nxt = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            pred.append(nxt)
            generated.append(nxt)
        correct += pred == a
        total += 1
        target_nll += s
        target_tokens += len(a)
    return {
        "accuracy": correct / total if total else float("nan"),
        "perplexity": math.exp(target_nll / target_tokens) if target_tokens else float("nan"),
        "total": total,
    }


def eval_piqa(model, limit):
    data = limited(ds("lighteval/piqa", split="validation"), limit)
    correct = total = 0
    for ex in tqdm(data, desc="PIQA"):
        losses = [mc_score(model, ex["goal"] + "\nAnswer:", " " + ex[k]) for k in ("sol1", "sol2")]
        correct += min(range(2), key=losses.__getitem__) == int(ex["label"])
        total += 1
    return {"accuracy": correct / total, "total": total}


def eval_openbookqa(model, limit):
    data = limited(ds("allenai/openbookqa", "main", "validation"), limit)
    correct = total = 0
    for ex in tqdm(data, desc="OpenBookQA"):
        texts = ex["choices"]["text"]
        labels = ex["choices"]["label"]
        losses = [mc_score(model, ex["question_stem"] + "\nAnswer:", " " + t) for t in texts]
        pred = min(range(len(losses)), key=losses.__getitem__)
        gold = labels.index(ex["answerKey"])
        correct += pred == gold
        total += 1
    return {"accuracy": correct / total, "total": total}


def eval_arc(model, limit, challenge):
    cfg = "ARC-Challenge" if challenge else "ARC-Easy"
    data = limited(ds("allenai/ai2_arc", cfg, "validation"), limit)
    correct = total = 0
    for ex in tqdm(data, desc=cfg):
        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]
        losses = [mc_score(model, ex["question"] + "\nAnswer:", " " + t) for t in texts]
        pred = min(range(len(losses)), key=losses.__getitem__)
        gold = labels.index(ex["answerKey"])
        correct += pred == gold
        total += 1
    return {"accuracy": correct / total, "total": total}


def eval_winogrande(model, limit):
    data = limited(ds("allenai/winogrande", "winogrande_xl", "validation"), limit)
    correct = total = 0
    for ex in tqdm(data, desc="WinoGrande"):
        prefix, suffix = ex["sentence"].split("_", 1)
        losses = []
        for key in ("option1", "option2"):
            c = prefix
            a = ex[key] + suffix
            losses.append(mc_score(model, c, a))
        pred = min(range(2), key=losses.__getitem__)
        correct += pred == int(ex["answer"]) - 1
        total += 1
    return {"accuracy": correct / total, "total": total}


def eval_boolq(model, limit):
    data = limited(ds("google/boolq", split="validation"), limit)
    correct = total = 0
    for ex in tqdm(data, desc="BoolQ"):
        prompt = f"Passage: {ex['passage']}\nQuestion: {ex['question']}\nAnswer:"
        scores = [mc_score(model, prompt, x) for x in (" yes", " no")]
        pred = scores[0] < scores[1]
        correct += pred == bool(ex["answer"])
        total += 1
    return {"accuracy": correct / total, "total": total}


def eval_rte(model, limit):
    data = limited(ds("nyu-mll/glue", "rte", "validation"), limit)
    correct = total = 0
    for ex in tqdm(data, desc="RTE"):
        prompt = f"Premise: {ex['sentence1']}\nHypothesis: {ex['sentence2']}\nAnswer:"
        scores = [mc_score(model, prompt, " entailment"), mc_score(model, prompt, " not entailment")]
        pred = min(range(2), key=scores.__getitem__)
        correct += pred == int(ex["label"])
        total += 1
    return {"accuracy": correct / total, "total": total}


def eval_wsc(model, limit):
    data = limited(ds("aps/super_glue", "wsc.fixed", "validation"), limit)
    correct = total = 0
    for ex in tqdm(data, desc="WSC"):
        prompt = f"Text: {ex['text']}\nQuestion: Does \"{ex['span1_text']}\" refer to \"{ex['span2_text']}\"?\nAnswer:"
        scores = [mc_score(model, prompt, " No"), mc_score(model, prompt, " Yes")]
        pred = 1 if scores[1] < scores[0] else 0
        correct += pred == int(ex["label"])
        total += 1
    return {"accuracy": correct / total, "total": total}


def eval_triviaqa(model, limit):
    data = limited(ds("mandarjoshi/trivia_qa", "rc.nocontext", "validation"), limit)
    correct = total = 0
    for ex in tqdm(data, desc="TriviaQA"):
        # Limita la respuesta a la primera línea generada
        pred = model.greedy_generate(f"Question: {ex['question']}\nAnswer:", 16).strip().split('\n')[0]
        refs = list(ex["answer"]["aliases"]) + [ex["answer"]["value"]]
        
        pred_norm = normalize_answer(pred)
        # Evaluamos por inclusión (substring) y no por coincidencia exacta
        ok = any(normalize_answer(r) in pred_norm for r in refs if r)
        correct += ok
        total += 1
    return {"accuracy": correct / total, "total": total}


def eval_arithmetic(model, limit, seed=42):
    rng = random.Random(seed)
    n = limit or 500
    correct = 0
    for _ in tqdm(range(n), desc="Arithmetic"):
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        op = rng.choice(("+", "*"))
        answer = a + b if op == "+" else a * b
        prompt = f"{a} {op} {b} ="
        generated = model.greedy_generate(
            prompt,
            max_new_tokens=max(8, len(str(answer)) + 2),
        ).strip()
        pred = generated.split()[0] if generated else ""
        correct += pred == str(answer)
    return {"accuracy": correct / n, "total": n}


def eval_storycloze(model, limit):
    data = limited(
        ds("MoE-UNC/story_cloze", split="validation"),
        limit
    )
    correct = total = 0
    for ex in tqdm(data, desc="StoryCloze"):
        ctx = (
            f"{ex['input_sentence_1']} "
            f"{ex['input_sentence_2']} "
            f"{ex['input_sentence_3']} "
            f"{ex['input_sentence_4']}"
        )

        losses = [
            mc_score(model,ctx," " + ex["sentence_quiz1"]),
            mc_score(model,ctx," " + ex["sentence_quiz2"]),
        ]

        pred = min(range(2), key=losses.__getitem__)
        gold = int(ex["answer_right_ending"]) - 1

        correct += pred == gold
        total += 1

    return {
        "accuracy": correct / total if total else 0.0,
        "total": total,
    }


def eval_webquestions(model, limit):
    data = limited(ds("stanfordnlp/web_questions", split="test"), limit)
    correct = total = 0
    for ex in tqdm(data, desc="WebQuestions"):
        prompt = f"Question: {ex['question']}\nAnswer:"
        pred = model.greedy_generate(prompt, max_new_tokens=16).strip().split('\n')[0]
        refs = ex.get("answers", [])
        
        pred_norm = normalize_answer(pred)
        # Evaluamos por inclusión (substring)
        ok = any(normalize_answer(r) in pred_norm for r in refs if r)
        correct += int(ok)
        total += 1
    return {"accuracy": correct / total if total else 0.0, "total": total}


def eval_copa(model, limit):
    data = limited(ds("aps/super_glue", "copa", "validation"), limit)
    correct = total = 0
    for ex in tqdm(data, desc="COPA"):
        connector = " because" if ex["question"] == "cause" else " so"
        prompt = ex["premise"] + connector
        losses = [
            mc_score(model, prompt, " " + ex["choice1"]),
            mc_score(model, prompt, " " + ex["choice2"]),
        ]
        pred = min(range(2), key=losses.__getitem__)
        correct += pred == int(ex["label"])
        total += 1
    return {"accuracy": correct / total if total else 0.0, "total": total}


def eval_race(model, limit):
    # Cambiado a 'ehovy/race'
    data = limited(ds("ehovy/race", "all", "validation"), limit)
    correct = total = 0
    mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
    for ex in tqdm(data, desc="RACE"):
        prompt = f"Article: {ex['article']}\nQuestion: {ex['question']}\nAnswer:"
        losses = [mc_score(model, prompt, " " + opt) for opt in ex["options"]]
        pred = min(range(len(losses)), key=losses.__getitem__)
        gold = mapping.get(str(ex["answer"]).strip().upper(), 0)
        correct += pred == gold
        total += 1
    return {"accuracy": correct / total if total else 0.0, "total": total}


def eval_mmlu(model, limit):
    data = limited(ds("cais/mmlu", "all", "test"), limit)
    correct = total = 0
    for ex in tqdm(data, desc="MMLU"):
        prompt = f"Question: {ex['question']}\nAnswer:"
        losses = [mc_score(model, prompt, " " + str(opt)) for opt in ex["choices"]]
        pred = min(range(len(losses)), key=losses.__getitem__)
        correct += pred == int(ex["answer"])
        total += 1
    return {"accuracy": correct / total if total else 0.0, "total": total}


def eval_gsm8k(model, limit):
    # Cambiado a 'openai/gsm8k'
    data = limited(ds("openai/gsm8k", "main", "test"), limit)
    correct = total = 0
    for ex in tqdm(data, desc="GSM8K"):
        prompt = f"Question: {ex['question']}\nAnswer:"
        generated = model.greedy_generate(prompt, max_new_tokens=64)
        target_str = ex["answer"].split("####")[-1].strip().replace(",", "")
        gen_numbers = re.findall(r"-?\d+(?:\.\d+)?", generated.replace(",", ""))
        pred = gen_numbers[-1] if gen_numbers else ""
        correct += pred == target_str
        total += 1
    return {"accuracy": correct / total if total else 0.0, "total": total}


def eval_truthfulqa(model, limit):
    data = limited(ds("truthfulqa/truthful_qa", "multiple_choice", "validation"), limit)
    correct = total = 0
    for ex in tqdm(data, desc="TruthfulQA"):
        prompt = f"Q: {ex['question']}\nA:"
        choices = ex["mc1_targets"]["choices"]
        labels = ex["mc1_targets"]["labels"]
        losses = [mc_score(model, prompt, " " + choice) for choice in choices]
        pred = min(range(len(losses)), key=losses.__getitem__)
        gold = labels.index(1) if 1 in labels else 0
        correct += pred == gold
        total += 1
    return {"accuracy": correct / total if total else 0.0, "total": total}


# =============================================================================
# Registry / CLI
# =============================================================================

TASKS = {
    "hellaswag": eval_hellaswag,
    "lambada": eval_lambada,
    "piqa": eval_piqa,
    "openbookqa": eval_openbookqa,
    "arc_easy": lambda m, l: eval_arc(m, l, False),
    "arc_challenge": lambda m, l: eval_arc(m, l, True),
    "winogrande": eval_winogrande,
    "boolq": eval_boolq,
    "rte": eval_rte,
    "wsc": eval_wsc,
    "triviaqa": eval_triviaqa,
    "arithmetic": eval_arithmetic,
    "storycloze": eval_storycloze,
    "webquestions": eval_webquestions,
    "copa": eval_copa,
    "race": eval_race,
    "mmlu": eval_mmlu,
    "gsm8k": eval_gsm8k,
    "truthfulqa": eval_truthfulqa,
}

DEFAULT_TASKS = [
    "hellaswag",
    "lambada",  
    "piqa",
    "openbookqa",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "wsc",
    "boolq",
    "rte",
    "triviaqa",
    "storycloze",
    "webquestions",
    "copa",
    "race",
    "mmlu",
    "gsm8k",
    "truthfulqa"
]

def validate_datasets(task_names):
    """Check that all selected benchmark datasets can be loaded."""
    checks = {
    "hellaswag": ("allenai/hellaswag", None, "validation"),
    "lambada": ("EleutherAI/lambada_openai", "default", "test"),
    "piqa": ("lighteval/piqa", None, "validation"),
    "openbookqa": ("allenai/openbookqa", "main", "validation"),
    "arc_easy": ("allenai/ai2_arc","ARC-Easy","validation"),
    "arc_challenge": ("allenai/ai2_arc","ARC-Challenge","validation"),
    "winogrande": ("allenai/winogrande","winogrande_xl","validation"),
    "wsc": ("aps/super_glue","wsc.fixed","validation"),
    "boolq": ("google/boolq",None,"validation"),
    "rte": ("nyu-mll/glue","rte","validation"),
    "triviaqa": ("mandarjoshi/trivia_qa","rc.nocontext","validation"),
    "storycloze": ("MoE-UNC/story_cloze",None,"validation"),
    "webquestions": ("stanfordnlp/web_questions",None,"test"),
    "copa": ("aps/super_glue","copa","validation"),
    "race": ("ehovy/race","all","validation"),
    "mmlu": ("cais/mmlu","all","test"),
    "gsm8k": ("openai/gsm8k","main","test"),
    "truthfulqa": ("truthfulqa/truthful_qa","multiple_choice","validation"),
    }

    for task in task_names:
        if task == "arithmetic" or task not in checks:
            continue

        path, name, split = checks[task]
        label = f"{path}" + (f" [{name}]" if name else "")
        print(f"Checking {task}: {label}")

        data = ds(path, name, split)

        if len(data) == 0:
            raise RuntimeError(f"{task} loaded but contains zero examples.")

        print(f"  OK - {len(data):,} examples")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", nargs="+", required=True, help="Checkpoint path(s) or HF model ID(s)")
    p.add_argument("--tasks", default=",".join(DEFAULT_TASKS), help="Comma-separated tasks or 'all'")
    p.add_argument("--limit", type=int, default=None, help="Maximum examples per task")
    p.add_argument("--device", default=None, help="cuda / cpu; auto if omitted")
    p.add_argument("--output", default="eval_results.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--check-datasets",
        action="store_true",
        help="Check/download selected datasets and exit.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    enc = tiktoken.get_encoding("gpt2")
    task_names = list(TASKS) if args.tasks.lower() == "all" else [x.strip() for x in args.tasks.split(",") if x.strip()]

    unknown = [x for x in task_names if x not in TASKS]
    if unknown:
        raise ValueError(
            f"Unknown tasks {unknown}. Available: {', '.join(TASKS)}"
        )

    if args.check_datasets:
        validate_datasets(task_names)
        print("\nAll selected datasets are accessible.")
        return

    results = {"meta": {"device": device, "tasks": task_names, "limit": args.limit, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, "models": {}}

    for identifier in args.model:
        print("\n" + "=" * 80)
        print(f"MODEL: {identifier}")
        print("=" * 80)
        model = load_model(identifier, device, enc)
        info = {
            "parameters": model.params,
            "parameters_millions": model.params / 1e6,
            "n_layer": getattr(model.config, "n_layer", None),
            "n_head": getattr(model.config, "n_head", None),
            "n_embd": getattr(model.config, "n_embd", None),
            "tasks": {},
        }
        print(f"Parameters: {model.params:,} ({model.params / 1e6:.2f}M)")
        for task in task_names:
            print(f"\n--- {task} ---")
            t0 = time.time()
            try:
                if task == "arithmetic":
                    r = eval_arithmetic(model, args.limit, args.seed)
                else:
                    r = TASKS[task](model, args.limit)
                r["seconds"] = time.time() - t0
                info["tasks"][task] = r
                print(json.dumps(r, indent=2))
            except Exception as exc:
                print(f"ERROR: {exc}")
                info["tasks"][task] = {"error": str(exc), "seconds": time.time() - t0}
        results["models"][model.name] = info
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"Results saved to: {out}")
    print("=" * 80)
    for name, info in results["models"].items():
        print(f"\n{name} ({info['parameters_millions']:.2f}M)")
        for task, r in info["tasks"].items():
            if "error" in r:
                print(f"  {task:16s} ERROR")
            else:
                bits = []
                if "accuracy" in r:
                    bits.append(f"acc={r['accuracy']:.4f}")
                if "accuracy_norm" in r:
                    bits.append(f"acc_norm={r['accuracy_norm']:.4f}")
                if "perplexity" in r:
                    bits.append(f"ppl={r['perplexity']:.4f}")
                print(f"  {task:16s} " + ", ".join(bits))


if __name__ == "__main__":
    main()