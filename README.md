# Decision model: Qwen3-1.7B + LoRA

Модель рішень на кшталт Jev з API `/v1/systemone`: текст + запитання з варіантами → ймовірність кожного
варіанта і впевненість. Підхід 2 з завдання: fine-tune LLM з LoRA.

## Вимоги завдання

| Вимога | Результат (v2) |
|---|---|
| noul (так/ні) | ✅ test acc 0.928, ECE 0.025 |
| choice (2–8 варіантів) | ✅ test acc 0.911, ECE 0.029 |
| \* score (шкала) | ✅ Spearman 0.88 на 15 текстах (`experiments/score_check.py`) |
| \* правильний формат | ✅ модель нічого не генерує, `output_tokens = 0` |
| \* калібрація | ✅ ECE ~0.03 проти ~0.2 без донавчання |

## Як працює

Запитання перетворюється на тест: `Text / Question / A) … B) … / Answer with one letter`. Модель читає його
один раз, і ми беремо шанси, з якими вона написала б кожну з літер: `P = softmax(оцінки літер / T)`.
Температура T (~1.05) підібрана на validation. noul — це варіанти Так/Ні; score — choice по рівнях,
`score = Σ i·pᵢ`.

LoRA (r = 16) навчає 17.4 млн параметрів, що становить 1% від Qwen; сама Qwen не змінюється.
Втрати — cross-entropy лише по літерах варіантів.

## Дані

Готові датасети з HF перетворено на запитання. Порядок і кількість варіантів випадкові, у noul 50/50
так/ні, ~40% запитань українською.

- **train (15 000):** banking77, ag_news, emotion, multi_nli (choice + noul), boolq, imdb (noul)
- **val / test:** 800 / 800 — інші тексти тих самих датасетів
- **OOD (1650):** clinc150, sms_spam, typed-decisions — датасети, яких у навчанні не було

**v1** — ноутбук, 938 кроків, 27 хв на GPU. **v2** — `train_v2.py`: ті самі дані + 1500 запитань
«чи це спам?» на e-mail (`SetFit/enron_spam`), 150 хв на M3 Pro. v2 потрібна, бо v1 провалила sms_spam:
вона боялася казати «так» і знаходила лише 34% спаму (`experiments/sms_analysis.py`).

## Результати

Метрики пораховані через API (`evaluate.py`); baseline — Qwen без адаптера.

| set | тип | acc base | acc v1 | acc v2 | ECE base | ECE v1 | ECE v2 |
|---|---|---:|---:|---:|---:|---:|---:|
| test | choice | 0.748 | 0.888 | **0.911** | 0.244 | 0.042 | **0.029** |
| test | noul | 0.819 | 0.916 | **0.928** | 0.179 | **0.018** | 0.025 |
| OOD | choice | 0.777 | 0.818 | **0.834** | 0.216 | **0.035** | 0.043 |
| OOD | noul | 0.850 | 0.771 | **0.874** | 0.145 | 0.080 | **0.047** |
| OOD sms_spam | noul | 0.890 | 0.668 | **0.892** | 0.102 | 0.131 | **0.054** |

**Калібрація (test):** baseline у 779 з 800 відповідей впевнений на >95%, а правий лише у 80% випадків.
Відповіді v2 з упевненістю >95% правильні у 99% випадків.

**Обмеження:** v2 позначає як спам 17% звичайних SMS (v1 — 0%). Найслабше місце — typed-decisions
(acc 0.58 choice / 0.72 noul).

Детально: `results/eval_summary.csv`, `results/calibration.csv`.

## Файли

| | |
|---|---|
| `decision_model_noul_choice.ipynb` | дані, навчання v1, метрики → `decision_lora_v1/` |
| `train_v2.py` | навчання v2 → `decision_lora_v2/` |
| `decision_model.py` | інференс: запит → відповідь systemone |
| `app.py` | FastAPI-сервер |
| `evaluate.py` | accuracy / ECE / токени → `results/` |
| `experiments/` | перевірка score, аналіз спаму |

## Запуск

```bash
pip install -r requirements.txt   # з uv: uv pip install --index-strategy unsafe-best-match -r requirements.txt
uvicorn app:app --port 8000        # v2; для v1: DECISION_ADAPTER=decision_lora_v1 uvicorn app:app --port 8000
```

Демо: http://localhost:8000/docs → `POST /v1/systemone` → **Try it out**. Там уже підставлено приклад
із завдання з усіма трьома типами запитань:

```json
"department":   {"type": "choice", "choice": "billing", "confidence": 0.9708, "probabilities": {...}},
"frustration":  {"type": "score", "score": 1.1299, "confidence": 0.6762, "probabilities": {...}},
"wants_refund": {"type": "noul", "noul": 0.9339, "confidence": 0.9339},
"usage": {"input_tokens": 581, "output_tokens": 0}
```

Відтворення: `python evaluate.py baseline|v1|v2` (~8 хв кожен), `python experiments/sms_analysis.py`,
`python experiments/score_check.py`, `python train_v2.py` (~2.5 год).
