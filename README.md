# Decision model (noul + choice)

Qwen3-1.7B + LoRA, що відповідає на запитання типу `choice` (вибір з 2–8 варіантів) і `noul` (так/ні) одним forward pass: варіанти позначаються літерами `A..H`, ймовірності — softmax логітів літер із підібраною температурою.

## Запуск

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Відкрити `decision_model_noul_choice.ipynb` з ядром `.venv` і виконати всі клітинки. Адаптер зберігається в `decision_lora/`.

## Результати (test / OOD)

| set  | qtype  | acc baseline | acc fine-tuned | ECE fine-tuned |
|------|--------|-------------:|---------------:|---------------:|
| test | choice | 0.748 | 0.891 | 0.037 |
| test | noul   | 0.809 | 0.916 | 0.015 |
| ood  | choice | 0.775 | 0.820 | 0.035 |
| ood  | noul   | 0.822 | 0.780 | 0.075 |

## API

`decision_model.py` — інференс (промпт, скоринг логітів літер, `systemone`): базова модель + адаптер з `decision_lora/` через `PeftModel.from_pretrained`, температура й промпт — з `decision_lora/decision_config.json`. Пристрій вибирається автоматично: CUDA → MPS → CPU.

```bash
uvicorn app:app --port 8000
```

Swagger: http://localhost:8000/docs

```bash
curl -X POST http://localhost:8000/v1/systemone -H "Content-Type: application/json" -d '{
  "state": "Вітаю! З мене двічі списали оплату за місячну підписку. Поверніть, будь ласка, зайве.",
  "questions": {
    "department":   {"type": "choice", "instructions": "Яка команда має обробити цей запит?",
                     "criteria": {"billing": "Рахунки, оплати, повернення коштів",
                                  "technical": "Баги, збої, помилки системи", "other": "Усе інше"}},
    "wants_refund": {"type": "noul", "instructions": "Клієнт явно просить повернути кошти?"}
  }
}'
```

Підтримуються типи `choice` (2–8 варіантів, `criteria` — словник або список) і `noul`; для інших типів у відповіді повертається `error`.
