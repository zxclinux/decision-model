"""Inference for the decision model (Qwen3-1.7B + LoRA), ported from decision_model_noul_choice.ipynb.

Each question is answered with one forward pass: options are labelled A..H and their probabilities are
softmax(letter logits / T), where T is the temperature fitted on the validation set.
"""
import json
import re
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ADAPTER_DIR = Path(__file__).resolve().parent / "decision_lora"
MODEL_TAG = "s1-qwen3-1.7b-lora"

with open(ADAPTER_DIR / "decision_config.json", encoding="utf-8") as f:
    CFG = json.load(f)

LETTERS = CFG["letters"]
SYSTEM_PROMPT = CFG["system_prompt"]
TEMPERATURE = CFG["temperature"]
MAX_STATE_TOKENS = CFG["max_state_tokens"]

YES_NO = {"en": ["Yes", "No"], "uk": ["Так", "Ні"]}
CYRILLIC = re.compile(r"[а-яіїєґ]", re.IGNORECASE)
_LOCK = threading.Lock()  # FastAPI runs sync endpoints in a thread pool; one forward pass at a time


def _device_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


@lru_cache(maxsize=1)
def load():
    """Load tokenizer, base model + LoRA adapter and letter token ids (once)."""
    device, dtype = _device_dtype()
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(CFG["base_model"], dtype=dtype).to(device)
    model = PeftModel.from_pretrained(base, ADAPTER_DIR).eval()

    letter_ids = []
    for L in LETTERS:
        ids = tokenizer.encode(L, add_special_tokens=False)
        assert len(ids) == 1, f"{L} -> {ids}"
        letter_ids.append(ids[0])
    return tokenizer, model, torch.tensor(letter_ids, device=device)


def truncate_text(text, max_tokens):
    tokenizer = load()[0]
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return text if len(ids) <= max_tokens else tokenizer.decode(ids[:max_tokens]) + " ..."


def build_prompt_ids(state, question, options, max_state_tokens=MAX_STATE_TOKENS):
    tokenizer = load()[0]
    opts = "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(options))
    user = (f"Text:\n{truncate_text(state, max_state_tokens)}\n\nQuestion: {question}\n\n"
            f"Options:\n{opts}\n\nAnswer with one letter.")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def letter_scores(model, letter_ids, input_ids, attention_mask):
    """Logits of the letter tokens at the last non-padding position (no full lm_head pass)."""
    base = model.get_base_model()
    hidden = base.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    last = attention_mask.sum(dim=1) - 1
    h = hidden[torch.arange(hidden.size(0), device=hidden.device), last]
    W = base.lm_head.weight[letter_ids]
    return h.float() @ W.float().T


@torch.no_grad()
def score(input_ids, n_opt):
    _, model, letter_ids = load()
    ids = torch.tensor([input_ids], device=model.device)
    with _LOCK:
        sc = letter_scores(model, letter_ids, ids, torch.ones_like(ids))
        return sc[0, :n_opt].cpu().numpy()


def softmax(z, T=1.0):
    z = np.asarray(z, dtype=np.float64) / T
    e = np.exp(z - z.max())
    return e / e.sum()


def option_probs(state, question, options, T):
    ids = build_prompt_ids(state, question, options)
    return softmax(score(ids, len(options)), T), len(ids)


def systemone(request, T=None):
    T = TEMPERATURE if T is None else T
    answers, input_tokens = {}, 0
    for name, q in request["questions"].items():
        qtype, instr = q.get("type"), q.get("instructions", "")
        if qtype == "noul":
            lang = "uk" if CYRILLIC.search(instr) else "en"
            p, n_tok = option_probs(request["state"], instr, YES_NO[lang], T)
            answers[name] = {"type": "noul", "noul": round(float(p[0]), 4)}
        elif qtype == "choice":
            crit = q.get("criteria") or []
            keys = list(crit)
            if not 2 <= len(keys) <= len(LETTERS):
                raise ValueError(f"{name}: choice needs 2..{len(LETTERS)} options, got {len(keys)}")
            options = [f"{k}: {v}" for k, v in crit.items()] if isinstance(crit, dict) else keys
            p, n_tok = option_probs(request["state"], instr, options, T)
            order = np.argsort(-p)
            answers[name] = {"type": "choice", "choice": keys[order[0]],
                             "probabilities": {keys[i]: round(float(p[i]), 4) for i in order},
                             "confidence": round(float(p.max()), 4)}
        else:
            answers[name] = {"type": qtype, "error": f"unsupported question type: {qtype}"}
            continue
        input_tokens += n_tok
    return {"model": MODEL_TAG, "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0}}
