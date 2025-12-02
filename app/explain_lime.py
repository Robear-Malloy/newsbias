from typing import List, Dict, Any, Tuple
import numpy as np
import torch, shap
from lime.lime_text import LimeTextExplainer

def _predict_proba(texts, tokenizer, model, device="cpu", max_length=512):
    enc = tokenizer(texts, padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs

def _normalize_abs(values: np.ndarray) -> np.ndarray:
    a = np.abs(values)
    m = a.max() if a.size and a.max() > 0 else 1.0
    return a / m

def explain_with_lime_spans(
    text: str,
    tokenizer,
    model,
    device: str,
    target_idx: int,
    k: int = 6,
    max_length: int = 512,
) -> List[Dict[str, Any]]:
    if not text or not text.strip():
        return []

    def predict(texts: List[str]) -> np.ndarray:
        return _predict_proba(texts, tokenizer, model, device, max_length)

    explainer = LimeTextExplainer()

    exp = explainer.explain_instance(
        text,
        predict,
        labels=[target_idx],
        num_features=max(k * 3, k)  
    )

    word_weights = exp.as_list(label=target_idx)  

    spans_raw = []
    for word, weight in word_weights:
        pattern = r"\b{}\b".format(re.escape(word))
        for m in re.finditer(pattern, text):
            start, end = m.span()
            spans_raw.append({
                "start": int(start),
                "end": int(end),
                "text": text[start:end],
                "value": float(weight),
            })
            if len(spans_raw) >= k:
                break
        if len(spans_raw) >= k:
            break

    if not spans_raw:
        return []

    mags = np.array([abs(s["value"]) for s in spans_raw], dtype=float)
    nz = _normalize_abs(mags)

    spans = []
    for s, sc in zip(spans_raw, nz):
        v = s["value"]
        spans.append({
            "start": s["start"],
            "end": s["end"],
            "text": s["text"],
            "score": float(sc),                 
            "value": float(v),                  
            "sign": int(np.sign(v)) if v != 0 else 0,
            "source": "lime",
        })

    spans.sort(key=lambda d: abs(d["value"]), reverse=True)
    return spans[:k]