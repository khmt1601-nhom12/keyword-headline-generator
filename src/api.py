from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

# =======================================================
# LOAD MODEL
# =======================================================

MODEL_PATH = "models/best_model"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("ĐANG LOAD MODEL...")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

model.to(device)
model.eval()

print("Device:", device)

# =======================================================
# FASTAPI
# =======================================================

app = FastAPI(
    title="Keyword → Headline Generator",
    version="1.0"
)

app.mount(
    "/static",
    StaticFiles(directory="demo/static"),
    name="static"
)

templates = Jinja2Templates(
    directory="demo/templates"
)

# =======================================================
# HOME PAGE
# =======================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request
        }
    )

# =======================================================
# REQUEST MODEL
# =======================================================

class InputData(BaseModel):
    keywords: str

# =======================================================
# PREDICT API
# =======================================================

@app.post("/predict")
async def predict(data: InputData):

    inputs = tokenizer(
        data.keywords,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=25,
            num_beams=5,
            no_repeat_ngram_size=2,  
            repetition_penalty=1.5,
            early_stopping=True
        )

    headline = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return {
        "keywords": data.keywords,
        "headline": headline
    }