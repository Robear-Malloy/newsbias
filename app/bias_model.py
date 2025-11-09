from typing import Dict, Any, List
import os, json, re
from pathlib import Path
from .explain_shap import explain_with_shap_spans


import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = ["Left", "Center", "Right"]

BASE_DIR = Path(__file__).resolve().parent

# HF repo id 
CKPT = os.getenv("BIAS_MODEL_NAME")

DEVICE = "cpu"

try:
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))
except Exception:
    pass

# detect whether CKPT is a local directory with model files
ckpt_path = Path(CKPT)
is_local_dir = ckpt_path.is_dir()

# if user explicitly points to a local path but it doesn't exist, fail with error
if os.getenv("BIAS_MODEL_NAME", "Halfbendy/qbias_model") and ckpt_path.is_absolute() and not is_local_dir:
    raise RuntimeError(f"Configured BIAS_MODEL_NAME points to a non-existent path: {CKPT}")

# tokenizer from base model
TOKENIZER_ID = os.getenv("BIAS_TOKENIZER_NAME", "google-bert/bert-base-cased")

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_ID,
    use_fast=True
)
# trained model
model = AutoModelForSequenceClassification.from_pretrained(
    CKPT,
    local_files_only=is_local_dir
)
# normalize label maps 
id2label = getattr(model.config, "id2label", {})
if id2label:
    norm = {int(i): str(l).title() for i, l in id2label.items()}
    model.config.id2label = norm
    model.config.label2id = {v: k for k, v in norm.items()}
else:
    # fallback if model lacks mapping
    model.config.id2label = {0: "Left", 1: "Center", 2: "Right"}
    model.config.label2id = {"Left": 0, "Center": 1, "Right": 2}

# move and eval
model.to(DEVICE).eval()

# temperature scaling
T = 1.0
temp_path = ckpt_path / "temperature.json" if is_local_dir else None
if temp_path and temp_path.exists():
    try:
        T = float(json.load(open(temp_path)).get("temperature", 1.0))
    except Exception:
        T = 1.0

_PATTERNS = [
    r"\b(flood|surge|invasion|weaponize|radical|extremist|witch hunt)\b",
    r"\b(soft on crime|open borders|tax-and-spend)\b",
    r"\b(war on [a-z]+|fearmongering)\b",
    r"\b(bipartisan|both parties|left[- ]wing|right[- ]wing|progressive|conservative|liberal)\b",
    r"\b(regulation|tax(?:ation)?|immigration|abortion|gun control|climate)\b",
]

def _spans(text: str, k: int = 6) -> List[Dict[str, Any]]:
    """these print out politically salient words from the article. doesn't affect the model, just adds extra context"""
    out: List[Dict[str, Any]] = []
    if not text:
        return out
    for pat in _PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            out.append({
                "text": text[m.start():m.end()],
                "start": m.start(),  
                "end": m.end(),
                "score": 0.15,
            })
    out.sort(key=lambda s: s["start"])
    return out[:k]

@torch.no_grad()
def classify(text: str, use_shap: bool = False) -> Dict[str, Any]:
    # Empty/short guard
    if not text or not text.strip():
        base_probs = {"Left": 0.33, "Center": 0.34, "Right": 0.33}
        return {
            "label": "Center",
            "confidence": 0.34,
            "probs": base_probs,
            "rationale_spans": [],
        }

    # encode
    enc = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    # forward
    logits = model(**enc).logits.squeeze(0)  

    # temperature scaling
    z = logits / T
    z = z - z.max()

    # probabilities
    probs_tensor = torch.softmax(z, dim=-1)     
    probs_list = probs_tensor.detach().cpu().tolist()

    # id2label mapping 
    cfg_id2label = getattr(model.config, "id2label", None)
    if cfg_id2label:
        id2label = {int(k): str(v) for k, v in cfg_id2label.items()}
    else:
        id2label = {0: "Left", 1: "Center", 2: "Right"}

    # predicted label + confidence
    conf, idx = torch.max(probs_tensor, dim=-1)
    label = id2label.get(int(idx), ["Left", "Center", "Right"][int(idx)])

    # full distribution by label
    probs_by_label = {id2label[i]: float(probs_list[i]) for i in range(len(probs_list))}

    # keyword spans with shap
    if use_shap:
        try:
            from .explain_shap import explain_with_shap_spans
            spans = explain_with_shap_spans(
                text=text,
                tokenizer=tokenizer,
                model=model,
                device=DEVICE,
                target_idx=int(idx),
                k=6,
                max_length=512,
                merge=True,
            )
        except Exception as e:
            print(f"[WARN] SHAP explanation failed: {e}")
            spans = []
    else:
        # Generate regex spans but *only* include basic fields
        spans = []
        for s in _spans(text, k=6):
            spans.append({
                "text": s["text"],
                "start": s["start"],
                "end": s["end"],
                # omit all SHAP-related fields
            })

    return {
        "label": label,
        "confidence": round(float(conf), 3),
        "probs": probs_by_label,
        "rationale_spans": spans,
    }

