"""Evaluate a model on the notebook's test and OOD sets through the real API path, systemone().

    python evaluate.py baseline   # Qwen3-1.7B without the adapter (zero-shot), T = 1
    python evaluate.py v1         # adapter decision_lora_v1  (notebook)
    python evaluate.py v2         # adapter decision_lora_v2  (train_v2.py)

Each run writes results/eval_rows_<model>.csv (one row per question) and then rebuilds, from all the rows
files present, the comparison tables:
    results/eval_summary.csv   accuracy and ECE per set / source / question type
    results/calibration.csv    stated confidence vs real accuracy (reliability table)

Test / OOD examples are rebuilt by executing the notebook's data cells (same seed -> the same examples).
~8 min per model on Apple MPS.
"""
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

MODELS = {"baseline": "decision_lora_v1", "v1": "decision_lora_v1", "v2": "decision_lora_v2"}
if len(sys.argv) != 2 or sys.argv[1] not in MODELS:
    sys.exit(f"usage: python evaluate.py {{{'|'.join(MODELS)}}}")
NAME = sys.argv[1]
os.environ["DECISION_ADAPTER"] = MODELS[NAME]  # must be set before decision_model is imported

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import decision_model as dm  # noqa: E402

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
DATA_CELLS = (2, 6, 8, 9, 10, 11)  # config, prompt, dataset specs, example builders, splits


def notebook_examples():
    nb = json.loads((ROOT / "decision_model_noul_choice.ipynb").read_text(encoding="utf-8"))
    src = {i: "".join(nb["cells"][i]["source"]) for i in DATA_CELLS}
    # train examples are built only to consume the same random numbers; don't tokenize them
    src[11] = (src[11]
               .replace('train_ex = encode(build_split(train=True), CFG["max_state_train"])',
                        "train_ex = build_split(train=True)")
               .replace('train_ds = Dataset.from_list([{k: e[k] for k in ("input_ids", "gold", "n_opt", "length")} '
                        'for e in train_ex])', ""))
    ns = {"tokenizer": dm.load()[0]}
    for i in DATA_CELLS:
        exec(compile(src[i], f"notebook_cell_{i}", "exec"), ns)
    return ns["test_ex"], ns["ood_ex"]


def to_request(e):
    if e["qtype"] == "noul":
        q = {"type": "noul", "instructions": e["question"]}
    else:
        q = {"type": "choice", "instructions": e["question"], "criteria": list(e["options"])}
    return {"state": e["state"], "questions": {"q": q}}


def ece_score(conf, correct, n_bins=15):  # same as the notebook
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=float)
    edges, ece = np.linspace(0, 1, n_bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return ece


def run():
    t0 = time.time()
    test_ex, ood_ex = notebook_examples()
    print(f"{NAME}: adapter {'off' if NAME == 'baseline' else dm.ADAPTER_DIR.name} | "
          f"test {len(test_ex)} | ood {len(ood_ex)}", flush=True)
    model = dm.load()[1]
    T = 1.0 if NAME == "baseline" else dm.TEMPERATURE
    rows, output_tokens = [], 0
    with model.disable_adapter() if NAME == "baseline" else nullcontext():
        for set_name, examples in (("test", test_ex), ("ood", ood_ex)):
            for j, e in enumerate(examples):
                resp = dm.systemone(to_request(e), T=T)
                ans, output_tokens = resp["answers"]["q"], output_tokens + resp["usage"]["output_tokens"]
                extra = {}
                if e["qtype"] == "noul":  # the API answers P(yes); gold: is the correct option Yes/Так?
                    gold_yes = e["options"][e["gold"]] in ("Yes", "Так")
                    p, gold = np.array([ans["noul"], 1 - ans["noul"]]), 0 if gold_yes else 1
                    extra = dict(gold_yes=gold_yes, p_yes=ans["noul"])
                else:
                    p, gold = np.array([ans["probabilities"][o.strip()] for o in e["options"]]), e["gold"]
                rows.append(dict(set=set_name, source=e["source"], qtype=e["qtype"], question=e["question"],
                                 n_opt=len(e["options"]), conf=float(p.max()), correct=int(p.argmax() == gold),
                                 nll=float(-np.log(p[gold] + 1e-12)), **extra))
                if (j + 1) % 200 == 0:
                    print(f"  {set_name} {j + 1}/{len(examples)}  ({time.time() - t0:.0f}s)", flush=True)
    RES.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RES / f"eval_rows_{NAME}.csv", index=False)
    print(f"{len(rows)} questions, generated output tokens: {output_tokens}, {time.time() - t0:.0f}s")


def summarize():
    runs = {m: pd.read_csv(RES / f"eval_rows_{m}.csv") for m in MODELS if (RES / f"eval_rows_{m}.csv").is_file()}
    summary, calib = {}, []
    for m, d in runs.items():
        groups = list(d.groupby(["set", "source", "qtype"])) + \
                 [((s, "ALL", q), g) for (s, q), g in d.groupby(["set", "qtype"])]
        for key, g in groups:
            summary.setdefault(key, {"n": len(g)}).update({f"acc_{m}": g.correct.mean(),
                                                           f"ece_{m}": ece_score(g.conf, g.correct)})
        d = d.assign(bin=pd.cut(d.conf, [0, .5, .6, .7, .8, .9, .95, 1.0]).astype(str))
        for (s, b), g in d.groupby(["set", "bin"], sort=True):
            calib.append(dict(model=m, set=s, confidence=b, n=len(g), mean_confidence=g.conf.mean(),
                              accuracy=g.correct.mean()))
    cols = ["n"] + [f"acc_{m}" for m in runs] + [f"ece_{m}" for m in runs]
    table = pd.DataFrame.from_dict(summary, orient="index")[cols]
    table.index.names = ["set", "source", "qtype"]
    table = table.sort_index().round(3)
    table.to_csv(RES / "eval_summary.csv")
    pd.DataFrame(calib).round(3).to_csv(RES / "calibration.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.to_string())


if __name__ == "__main__":
    run()
    summarize()
