"""Experiment: why does v1 lose accuracy on the OOD set sms_spam (baseline 0.855 -> v1 0.68), and does v2 fix it?

Uses results/eval_rows_<model>.csv from evaluate.py (no model needed).

    python experiments/sms_analysis.py      ->  results/sms_analysis.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results"


def analyze(model):
    d = pd.read_csv(RES / f"eval_rows_{model}.csv")
    s = d[d.source == "sms_spam"].copy()
    s["gold_yes"] = s.gold_yes.astype(str).eq("True")
    s["pred_yes"] = s.p_yes > 0.5
    y, p = s.gold_yes.values, s.p_yes.values
    pos, neg = p[y], p[~y]
    auc = np.mean((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :]))
    lines = [f"=== {model}: n = {len(s)}, spam share = {y.mean():.2f}, accuracy = {s.correct.mean():.3f}", "",
             "confusion (rows: gold spam, cols: predicted spam)", pd.crosstab(s.gold_yes, s.pred_yes).to_string(), "",
             f"mean P(spam): spam {pos.mean():.3f} | ham {neg.mean():.3f}",
             f"ROC AUC of P(spam): {auc:.3f}   (how well P(spam) separates spam from ham at any threshold)", "",
             "by question:"]
    g = [dict(question=q, n=len(x), acc=x.correct.mean(), spam_recall=x[x.gold_yes].pred_yes.mean(),
              ham_false_pos=x[~x.gold_yes].pred_yes.mean()) for q, x in s.groupby("question")]
    lines += [pd.DataFrame(g).round(3).to_string(index=False), "", "decision threshold on P(spam):"]
    for t in (0.5, 0.3, 0.2, 0.1):
        pr = p > t
        lines.append(f"  {t:4}: acc {np.mean(pr == y):.3f}  spam recall {pr[y].mean():.3f}  "
                     f"ham false pos {pr[~y].mean():.3f}")
    return "\n".join(lines)


def main():
    text = "\n\n".join(analyze(m) for m in ("v1", "v2") if (RES / f"eval_rows_{m}.csv").is_file())
    print(text)
    (RES / "sms_analysis.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
