"""Experiment: is the zero-shot `score` type meaningful? (the adapter was trained on noul/choice only)

For hand-written texts with a known level we check
  * ordering: Spearman correlation between the true level and the predicted score;
  * position bias: the same levels listed in reverse order (highest first), mapped back to the original
    scale. If the model reads the level descriptions rather than their position, both scores agree.

    python experiments/score_check.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import decision_model as dm  # noqa: E402

SCALES = {
    "frustration": dict(
        instructions="Наскільки клієнт роздратований?",
        levels=["Спокійний, нейтральний тон", "Помірне невдоволення", "Сильне роздратування або погрози піти"],
        texts=[(0, "Добрий день! Підкажіть, будь ласка, як змінити адресу доставки в акаунті?"),
               (0, "Hi, could you tell me when the new plan becomes available? Thanks!"),
               (1, "Вже другий раз замовлення приходить із запізненням. Неприємно, сподіваюсь, виправите."),
               (1, "I'm a bit disappointed, the app keeps logging me out. Please look into it."),
               (2, "Це просто знущання! Третій тиждень без інтернету, ніхто не відповідає. Розриваю договір!"),
               (2, "This is unacceptable. I've been charged twice and nobody answers. I'm cancelling today.")]),
    "urgency": dict(
        instructions="How urgent is this request?",
        levels=["Not urgent, can wait", "Should be handled today", "Critical, needs immediate action"],
        texts=[(0, "When you have time, could you update the logo on the about page?"),
               (1, "The export to CSV is broken for some users, please fix it today if possible."),
               (2, "Production is down, all customers get error 500 on checkout right now!"),
               (2, "Сервер платежів не працює, усі оплати падають просто зараз!")]),
    "review": dict(
        instructions="Оцініть тональність відгуку.",
        levels=["Дуже негативний", "Негативний", "Нейтральний", "Позитивний", "Дуже позитивний"],
        texts=[(0, "Жахливий сервіс, гроші на вітер, нікому не раджу."),
               (1, "Доставка затрималась, і товар трохи подряпаний. Не дуже."),
               (2, "Товар відповідає опису. Нічого особливого."),
               (3, "Хороша якість, швидко доставили, задоволений."),
               (4, "Неймовірно! Найкраща покупка цього року, усім раджу!")]),
}


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    rows = []
    for scale, spec in SCALES.items():
        K = len(spec["levels"])
        for level, text in spec["texts"]:
            fwd = dm.systemone({"state": text, "questions": {"q": {
                "type": "score", "instructions": spec["instructions"], "criteria": spec["levels"]}}})["answers"]["q"]
            rev = dm.systemone({"state": text, "questions": {"q": {
                "type": "score", "instructions": spec["instructions"],
                "criteria": spec["levels"][::-1]}}})["answers"]["q"]
            rows.append(dict(scale=scale, K=K, level=level, score=fwd["score"],
                             score_rev=(K - 1) - rev["score"],  # map reversed scale back
                             argmax=int(max(fwd["probabilities"], key=fwd["probabilities"].get)),
                             conf=fwd["confidence"], text=text[:45]))

    print(f"{'scale':12s} {'true':>4s} {'score':>6s} {'rev→':>6s} {'argmax':>6s} {'conf':>5s}  text")
    for r in rows:
        print(f"{r['scale']:12s} {r['level']:4d} {r['score']:6.2f} {r['score_rev']:6.2f} {r['argmax']:6d} "
              f"{r['conf']:5.2f}  {r['text']}")

    lv, sc, sr = (np.array([r[k] for r in rows], dtype=float) for k in ("level", "score", "score_rev"))
    norm = np.array([r["K"] - 1 for r in rows], dtype=float)
    summary = dict(
        n=len(rows),
        exact_level_acc=float(np.mean([r["argmax"] == r["level"] for r in rows])),
        spearman_true_vs_score=spearman(lv / norm, sc / norm),
        mae_levels=float(np.mean(np.abs(sc - lv))),
        mean_abs_fwd_minus_rev=float(np.mean(np.abs(sc - sr))),
    )
    print(json.dumps(summary, indent=2))
    out = Path(__file__).resolve().parent.parent / "results"
    out.mkdir(exist_ok=True)
    (out / "score_check.json").write_text(json.dumps(dict(summary=summary, rows=rows), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
