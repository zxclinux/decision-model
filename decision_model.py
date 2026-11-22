"""Inference for the decision model (Qwen3-1.7B + LoRA adapter).

Every question becomes a multiple-choice test with options labelled A..H. The model reads it once
(one forward pass, nothing is generated) and we take the logits of the option letters at the last position:
P(option) = softmax(letter logits / T), where T is the temperature fitted on the validation set.

Adapter: decision_lora_v2 by default, DECISION_ADAPTER=decision_lora_v1 selects v1.
"""
import json
import os
import re
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
ADAPTER_DIR = ROOT / os.environ.get("DECISION_ADAPTER", "decision_lora_v2")
CONFIG_PATH = ADAPTER_DIR / "decision_config.json"
if not CONFIG_PATH.is_file():
    raise FileNotFoundError(f"{CONFIG_PATH} not found: put the trained adapter folder into the repo root")

with open(CONFIG_PATH, encoding="utf-8") as f:
    CFG = json.load(f)

LETTERS = CFG["letters"]                      # A..H -> at most 8 options
SYSTEM_PROMPT = CFG["system_prompt"]
TEMPERATURE = CFG["temperature"]
MAX_STATE_TOKENS = CFG["max_state_tokens"]    # longer texts are truncated
MAX_PROMPT_TOKENS = 1280                      # same limit as in training / evaluation
MODEL_TAG = f"s1-qwen3-1.7b-lora-{'v2' if ADAPTER_DIR.name.endswith('v2') else 'v1'}"

YES_NO = {"en": ["Yes", "No"], "uk": ["Так", "Ні"]}
CYRILLIC = re.compile(r"[а-яіїєґ]", re.IGNORECASE)
_LOCK = threading.Lock()  # FastAPI serves requests in a thread pool; run one forward pass at a time


@lru_cache(maxsize=1)
def load():
    """Load tokenizer, base model + LoRA adapter and the token ids of the letters (once)."""
    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        device, dtype = "cpu", torch.float32
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    base = AutoModelForCausalLM.from_pretrained(CFG["base_model"], dtype=dtype).to(device)
    model = PeftModel.from_pretrained(base, ADAPTER_DIR).eval()

    letter_ids = []
    for L in LETTERS:
        ids = tokenizer.encode(L, add_special_tokens=False)
        assert len(ids) == 1, f"{L} -> {ids}"  # every letter must be a single token
        letter_ids.append(ids[0])
    return tokenizer, model, torch.tensor(letter_ids, device=device)


def truncate_text(text, max_tokens):
    tokenizer = load()[0]
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return text if len(ids) <= max_tokens else tokenizer.decode(ids[:max_tokens]) + " ..."


def build_prompt_ids(state, question, options, max_state_tokens=MAX_STATE_TOKENS):
    """The same prompt as in training: text, question, lettered options, 'Answer with one letter.'"""
    tokenizer = load()[0]
    opts = "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(options))
    user = (f"Text:\n{truncate_text(state, max_state_tokens)}\n\nQuestion: {question}\n\n"
            f"Options:\n{opts}\n\nAnswer with one letter.")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


@torch.no_grad()
def score(input_ids, n_opt):
    """One forward pass -> logits of the first n_opt letters at the last position."""
    _, model, letter_ids = load()
    base = model.get_base_model()
    ids = torch.tensor([input_ids], device=model.device)
    with _LOCK:
        h = base.model(input_ids=ids).last_hidden_state[0, -1]
        W = base.lm_head.weight[letter_ids[:n_opt]]  # only the letter rows of the output layer
        return (h.float() @ W.float().T).cpu().numpy()


def softmax(z, T=1.0):
    z = np.asarray(z, dtype=np.float64) / T
    e = np.exp(z - z.max())
    return e / e.sum()


def prepare_question(name, q, state):
    """Validate one question and build its prompt. Raises ValueError before any model work is done.

    noul   -> options Yes/No (Так/Ні for Cyrillic instructions)
    choice -> criteria dict {key: description} or list of keys
    score  -> criteria list of ordered levels (0, 1, 2, ...); answered as a choice over the levels
    """
    qtype = q.get("type")
    instr = (q.get("instructions") or "").strip()
    if qtype not in ("noul", "choice", "score"):
        return None
    if not instr:
        raise ValueError(f"{name}: instructions must not be empty")
    crit, levels = q.get("criteria") or [], None
    if qtype == "noul":
        keys = None
        options = YES_NO["uk" if CYRILLIC.search(instr) else "en"]
    else:
        if qtype == "score":
            if isinstance(crit, dict):
                raise ValueError(f"{name}: score criteria must be a list of levels, lowest first")
            levels = [str(c).strip() for c in crit]
            keys = [str(i) for i in range(len(levels))]
        else:
            keys = [str(k).strip() for k in crit]
        if not 2 <= len(keys) <= len(LETTERS):
            raise ValueError(f"{name}: {qtype} needs 2..{len(LETTERS)} options, got {len(keys)}")
        if qtype == "score":
            if not all(levels):
                raise ValueError(f"{name}: levels must not be empty")
            options = [f"{k}: {v}" for k, v in zip(keys, levels)]
        else:
            if not all(keys):
                raise ValueError(f"{name}: option names must not be empty")
            if len(set(keys)) != len(keys):
                raise ValueError(f"{name}: option names must be unique")
            options = [f"{k}: {v}" for k, v in zip(keys, crit.values())] if isinstance(crit, dict) else keys
    ids = build_prompt_ids(state, instr, options)
    if len(ids) > MAX_PROMPT_TOKENS:
        raise ValueError(f"{name}: prompt is {len(ids)} tokens, limit is {MAX_PROMPT_TOKENS} "
                         f"(the text is truncated to {MAX_STATE_TOKENS} tokens; shorten instructions/options)")
    return dict(type=qtype, keys=keys, ids=ids, n_opt=len(options), legend=levels)


def make_answer(prep, p):
    """Turn option probabilities into the /v1/systemone answer for one question."""
    if prep["type"] == "noul":
        return {"type": "noul", "noul": round(float(p[0]), 4),
                "confidence": round(float(max(p[0], p[1])), 4)}
    keys = prep["keys"]
    if prep["type"] == "score":
        return {"type": "score",
                "score": round(float(np.dot(np.arange(len(p)), p)), 4),  # expected level
                "confidence": round(float(p.max()), 4),
                "probabilities": {k: round(float(pi), 4) for k, pi in zip(keys, p)},
                "legend": dict(zip(keys, prep["legend"]))}
    order = np.argsort(-p)
    return {"type": "choice", "choice": keys[order[0]],
            "probabilities": {keys[i]: round(float(p[i]), 4) for i in order},
            "confidence": round(float(p.max()), 4)}


def systemone(request, T=None):
    """Answer all questions of a /v1/systemone request (one forward pass per question)."""
    T = TEMPERATURE if T is None else T
    state = request.get("state") or ""
    if not state.strip():
        raise ValueError("state must not be empty")

    # 1) validate everything and build prompts, 2) only then run the model
    prepared = {name: prepare_question(name, q, state) for name, q in request["questions"].items()}
    answers, input_tokens = {}, 0
    for name, prep in prepared.items():
        if prep is None:
            qtype = request["questions"][name].get("type")
            answers[name] = {"type": qtype, "error": f"unsupported question type: {qtype}"}
            continue
        p = softmax(score(prep["ids"], prep["n_opt"]), T)
        answers[name] = make_answer(prep, p)
        input_tokens += len(prep["ids"])
    return {"model": MODEL_TAG, "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0}}
