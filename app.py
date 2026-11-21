"""FastAPI server: POST /v1/systemone.  Run: uvicorn app:app --port 8000"""
from contextlib import asynccontextmanager
from typing import Dict, List, Literal, Optional, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import decision_model


class Question(BaseModel):
    type: Literal["choice", "noul", "score"]
    instructions: str = ""
    criteria: Optional[Union[Dict[str, str], List[str]]] = None


class SystemOneRequest(BaseModel):
    model: Optional[str] = None
    state: str
    questions: Dict[str, Question] = Field(min_length=1)

    model_config = {"json_schema_extra": {"examples": [{
        "model": "s1-qwen3-1.7b-lora",
        "state": "Вітаю! З мене двічі списали оплату за місячну підписку. Поверніть, будь ласка, зайве. "
                 "Акаунт працює нормально, але я вже третій день не можу до вас додзвонитися.",
        "questions": {
            "department": {"type": "choice", "instructions": "Яка команда має обробити цей запит?",
                           "criteria": {"billing": "Рахунки, оплати, повернення коштів",
                                        "technical": "Баги, збої, помилки системи",
                                        "sales": "Ціни, нові контракти",
                                        "other": "Усе інше"}},
            "wants_refund": {"type": "noul", "instructions": "Клієнт явно просить повернути кошти?"},
        },
    }]}}


@asynccontextmanager
async def lifespan(app):
    decision_model.load()  # load the model once at startup, not on the first request
    yield


app = FastAPI(title="Decision model", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": decision_model.MODEL_TAG}


@app.post("/v1/systemone")
def systemone(request: SystemOneRequest):
    try:
        return decision_model.systemone(request.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
