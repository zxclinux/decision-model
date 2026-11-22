"""Train adapter v2: the notebook's training data + 1500 yes/no questions about e-mail spam (SetFit/enron_spam).

Why: v1 fails on the OOD set sms_spam (acc 0.68, see results/sms_analysis.txt) because there is no spam
in its training data. sms_spam itself stays out of training, so it remains an honest OOD check.

The notebook's data cells are executed as-is, so train / val / test / OOD are exactly the notebook's examples
(the spam examples use their own random generator). Model, LoRA and training hyper-parameters are the
notebook's CFG. Runs on CUDA or Apple MPS (~2.5 h on M3 Pro).

    python train_v2.py      ->  decision_lora_v2/  and  results/train_v2_loss.csv
"""
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "decision_lora_v2"
N_SPAM, SPAM_UK_SHARE = 1500, 0.5
DATA_CELLS = (2, 6, 8, 9, 10, 11)  # config, prompt, dataset specs, example builders, splits
SPAM_Q = {"en": ["Is this email spam?", "Is this an unsolicited commercial or phishing email?",
                 "Is this message junk mail?"],
          "uk": ["Чи цей лист — спам?", "Чи це небажана реклама або фішинговий лист?",
                 "Чи це повідомлення — рекламна розсилка або шахрайство?"]}


def run_cells(ns, nb, cells):
    for i in cells:
        exec(compile("".join(nb["cells"][i]["source"]), f"notebook_cell_{i}", "exec"), ns)


def notebook_data(nb):
    """Execute the notebook's data cells: returns their namespace (CFG, train_ex, val_ex, ...)."""
    ns = {}
    run_cells(ns, nb, DATA_CELLS[:1])
    ns["tokenizer"] = AutoTokenizer.from_pretrained(ns["CFG"]["model_name"])
    if ns["tokenizer"].pad_token is None:
        ns["tokenizer"].pad_token = ns["tokenizer"].eos_token
    run_cells(ns, nb, DATA_CELLS[1:])
    return ns


def spam_examples(ns, n, uk_share, seed=2026):
    """Balanced yes/no 'is this spam' questions on e-mails, built like the notebook's make_noul()."""
    rng = random.Random(seed)
    ds = load_dataset("SetFit/enron_spam", split="train").shuffle(seed=seed).select(range(20 * n))
    pos = [r for r in ds if r["label"] == 1][: n // 2]
    neg = [r for r in ds if r["label"] == 0][: n - n // 2]
    out = []
    for r in pos + neg:
        lang = "uk" if rng.random() < uk_share else "en"
        options, gold = list(ns["YES_NO"][lang]), 0 if r["label"] == 1 else 1
        if rng.random() < 0.5:  # random option order, as in the notebook
            options.reverse(); gold = 1 - gold
        out.append(dict(state=r["text"], question=rng.choice(SPAM_Q[lang]), options=options, gold=gold,
                        qtype="noul", source="enron_spam"))
    rng.shuffle(out)
    return out


def main():
    t_start = time.time()
    nb = json.loads((ROOT / "decision_model_noul_choice.ipynb").read_text(encoding="utf-8"))
    ns = notebook_data(nb)
    CFG, tokenizer = ns["CFG"], ns["tokenizer"]
    extra = ns["encode"](spam_examples(ns, N_SPAM, SPAM_UK_SHARE), CFG["max_state_train"])
    train_ex = ns["train_ex"] + extra
    random.Random(7).shuffle(train_ex)
    print(f"train {len(ns['train_ex'])} + spam {len(extra)} = {len(train_ex)} | val {len(ns['val_ex'])}")
    train_ds = Dataset.from_list([{k: e[k] for k in ("input_ids", "gold", "n_opt", "length")} for e in train_ex])

    # base model + LoRA, as in the notebook
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16  # base weights; the LoRA weights themselves stay fp32
    model = AutoModelForCausalLM.from_pretrained(CFG["model_name"], dtype=dtype).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", bias="none", **CFG["lora"]))
    model.print_trainable_parameters()
    ns.update(model=model, DEVICE=device, DTYPE=dtype)
    run_cells(ns, nb, (13,))  # letter_scores, collate, score_examples
    letter_scores, collate = ns["letter_scores"], ns["collate"]

    class LetterTrainer(Trainer):  # the notebook's loss: cross-entropy over the option letters only
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            scores = letter_scores(model, inputs["input_ids"], inputs["attention_mask"])
            n_opt = inputs["n_opt"]
            scores = scores.masked_fill(torch.arange(scores.size(1), device=scores.device)[None, :] >= n_opt[:, None],
                                        float("-inf"))
            loss = F.cross_entropy(scores, inputs["gold"])
            return (loss, scores) if return_outputs else loss

    args = TrainingArguments(output_dir=str(ROOT / "outputs"), optim="adamw_torch", bf16=False, fp16=False,
                             save_strategy="no", report_to="none", train_sampling_strategy="group_by_length",
                             remove_unused_columns=False, label_names=["gold"], seed=CFG["seed"],
                             dataloader_pin_memory=False, **CFG["train"])
    trainer = LetterTrainer(model=model, args=args, train_dataset=train_ds, data_collator=collate)
    t0 = time.time()
    trainer.train()
    train_min = (time.time() - t0) / 60
    print(f"trained {trainer.state.global_step} steps in {train_min:.1f} min")

    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "train_v2_loss.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "loss", "learning_rate"])
        for r in trainer.state.log_history:
            if "loss" in r:
                w.writerow([r["step"], round(r["loss"], 4), r["learning_rate"]])

    # temperature T: minimise NLL on the validation set, as in the notebook
    val_ex = ns["val_ex"]
    val_logits = ns["score_examples"](val_ex)
    softmax = lambda z, T: np.exp(z / T - (z / T).max()) / np.exp(z / T - (z / T).max()).sum()
    T_grid = np.exp(np.linspace(np.log(0.1), np.log(10.0), 161))
    nll = [np.mean([-np.log(softmax(np.asarray(z, float), T)[e["gold"]] + 1e-12)
                    for z, e in zip(val_logits, val_ex)]) for T in T_grid]
    T_star = float(T_grid[int(np.argmin(nll))])
    acc = float(np.mean([np.argmax(z) == e["gold"] for z, e in zip(val_logits, val_ex)]))
    print(f"T* = {T_star:.3f}  (val acc {acc:.4f}, NLL {min(nll):.4f})")

    model.save_pretrained(OUT)
    tokenizer.save_pretrained(OUT)
    (OUT / "decision_config.json").write_text(json.dumps(
        {"base_model": CFG["model_name"], "temperature": T_star, "letters": ns["LETTERS"],
         "system_prompt": ns["SYSTEM_PROMPT"], "max_state_tokens": CFG["max_state_eval"],
         "train_extra": {"enron_spam_noul": len(extra), "spam_uk_share": SPAM_UK_SHARE},
         "train_steps": trainer.state.global_step, "train_minutes": round(train_min, 1),
         "train_device": str(device)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved to {OUT}  (total {(time.time() - t_start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
