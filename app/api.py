from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query
from .news_fetch import extract_article
from typing import List
from datetime import datetime
import os, json, logging
from dotenv import load_dotenv
load_dotenv()  

from .schemas import (
    PredictRequest, PredictResponse,
    EnvResponse, BiasOut, ExplainOut, RationaleSpan
)
from .summarizer import summarize
from .bias_model import classify

log = logging.getLogger("uvicorn.error")

app = FastAPI(title="Bias Classifier API", version="1.0.0")

# CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)

# -------- Model  ----------
CKPT_DIR  = os.getenv("BIAS_MODEL_NAME", "Halfbendy/qbias_model")
TEMP_PATH = os.path.join(CKPT_DIR, "temperature.json")
EVAL_PATH = os.path.join(CKPT_DIR, "eval_summary.json")

MODEL_INFO = {
    "model_name": os.path.basename(CKPT_DIR.rstrip("/")),
    "labels": ["Left","Center","Right"],
    "temperature": None, "val_metrics": {}
}

try:
    from .priors import get_prior_for_source  
except Exception:
    get_prior_for_source = None

if os.path.exists(TEMP_PATH):
    try:
        MODEL_INFO["temperature"] = json.load(open(TEMP_PATH)).get("temperature")
    except Exception as e:
        log.warning(f"temperature.json parse error: {e}")
if os.path.exists(EVAL_PATH):
    try:
        MODEL_INFO["val_metrics"] = json.load(open(EVAL_PATH))
    except Exception as e:
        log.warning(f"eval_summary.json parse error: {e}")

# -------------------- Endpoints --------------------
from .summarizer import USE_LLM as SUM_LLM, SUMMARY_MODEL as SUM_MODEL

@app.get("/env_debug")
def env_debug():
    import torch
    return {
        "device": "cpu",
        "torch": torch.__version__,
        "summary_llm": SUM_LLM,
        "summary_model": SUM_MODEL,
    }
    
@app.get("/healthz")
def healthz():
    return {"ok": True, "time": datetime.utcnow().isoformat() + "Z"}

@app.get("/env", response_model=EnvResponse)
def env():
    import torch
    return EnvResponse(
        device= "cpu",
        torch=torch.__version__,
        cuda_available=bool(torch.cuda.is_available()),
        cuda_version=getattr(torch.version, "cuda", None),
    )

@app.get("/model")
def model_info():
    """ Minimal model card for demo slides """
    return MODEL_INFO

@app.get("/fetch", tags=["ingest"], summary="Fetch & extract article content by URL")
async def fetch(url: str = Query(..., description="Article URL")):
    art = await extract_article(url)
    return art

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest = Body(...)):
    if not payload.text or len(payload.text) < 20:
        raise HTTPException(status_code=400, detail="`text` must be at least 20 characters.")

    full_text = ((payload.title or "").strip() + "\n\n" + (payload.text or "").strip()).strip()
    summ = summarize(payload.text or "").get("text", "")

    # bias model
    try:
        res = classify(full_text)
    except Exception as e:
        log.exception("classification error")
        raise HTTPException(status_code=500, detail=f"classifier failed: {e}")

    spans = [RationaleSpan(**s) for s in res.get("rationale_spans", [])]
    return PredictResponse(
        summary=summ,
        bias=BiasOut(
            label=res["label"],
            confidence=float(res["confidence"]),
            probs={k: round(v, 3) for k, v in (res.get("probs") or {}).items()}
        ),
        explain=ExplainOut(spans=spans)
    )

@app.post("/predict_url", response_model=PredictResponse)
async def predict_url(url: str = Body(..., embed=True)):
    art = await extract_article(url)    # {'url','source','title','text'}
    text = (art.get("text") or "")[:8000]
    if len(text) < 20:
        raise HTTPException(status_code=400, detail="Extracted text too short.")

    summ = summarize(text).get("text", "")
    full_text = ((art.get("title") or "").strip() + "\n\n" + text).strip()
    res = classify(full_text)
    spans = [RationaleSpan(**s) for s in res.get("rationale_spans", [])]
    # source-level prior from AllSides mapping
    prior = None
    if get_prior_for_source:
        # extractor returns domain like "cnn.com"; priors may be keyed by outlet name
        # try domain first, then strip www., then title-based guess
        key = (art.get("source") or "").lower()  # e.g., "cnn.com"
        try:
            prior = get_prior_for_source(key)  # implement domain to prior mapping in priors.py
        except Exception:
            prior = None

    return PredictResponse(
        summary=summ,
        bias=BiasOut(
            label=res["label"],
            confidence=float(res["confidence"]),
            probs={k: round(v, 3) for k, v in (res.get("probs") or {}).items()}
        ),
        explain=ExplainOut(spans=spans),
        source_prior=prior,
    )

@app.post("/predict_url_shap", response_model=PredictResponse)
async def predict_url_shap(url: str = Body(..., embed=True)):
    art = await extract_article(url)    # {'url','source','title','text'}
    text = (art.get("text") or "")[:8000]
    if len(text) < 20:
        raise HTTPException(status_code=400, detail="Extracted text too short.")

    summ = summarize(text).get("text", "")
    full_text = ((art.get("title") or "").strip() + "\n\n" + text).strip()
    res = classify(full_text, True)
    spans = [RationaleSpan(**s) for s in res.get("rationale_spans", [])]
    # source-level prior from AllSides mapping
    prior = None
    if get_prior_for_source:
        # extractor returns domain like "cnn.com"; priors may be keyed by outlet name
        # try domain first, then strip www., then title-based guess
        key = (art.get("source") or "").lower()  # e.g., "cnn.com"
        try:
            prior = get_prior_for_source(key)  # implement domain to prior mapping in priors.py
        except Exception:
            prior = None

    return PredictResponse(
        summary=summ,
        bias=BiasOut(
            label=res["label"],
            confidence=float(res["confidence"]),
            probs={k: round(v, 3) for k, v in (res.get("probs") or {}).items()}
        ),
        explain=ExplainOut(spans=spans),
        source_prior=prior,
    )

@app.post("/batch_predict", response_model=List[PredictResponse])
def batch_predict(items: List[PredictRequest] = Body(...)):
    outputs: List[PredictResponse] = []
    for item in items:
        full_text = ((item.title or "").strip() + "\n\n" + (item.text or "").strip()).strip()
        summ = summarize(item.text or "").get("text", "")
        res = classify(full_text)
        spans = [RationaleSpan(**s) for s in res.get("rationale_spans", [])]
        outputs.append(PredictResponse(
        summary=summ,
        bias=BiasOut(
            label=res["label"],
            confidence=float(res["confidence"]),
            probs={k: round(v, 3) for k, v in (res.get("probs") or {}).items()}
        ),
        explain=ExplainOut(spans=spans)
    ))
    return outputs
