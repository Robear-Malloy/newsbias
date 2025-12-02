from typing import List, Dict, Any
import numpy as np
import torch
import shap

def predict_proba(texts, tokenizer, model, device: str = "cpu", max_length: int = 512):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    return probs


def normalize_abs(values: np.ndarray) -> np.ndarray:
    a = np.abs(values)
    m = a.max() if a.size and a.max() > 0 else 1.0
    return a / m


def merge_adjacent_tokens(tokens, offsets, values):
    out = []
    if not tokens:
        return out

    cur = {
        "start": offsets[0][0],
        "end":   offsets[0][1],
        "text":  None,
        "value": float(values[0]),
        "tok_spans": [offsets[0]],
    }

    last_end = offsets[0][1]
    last_sign = np.sign(values[0])

    for i in range(1, len(tokens)):
        s, e = offsets[i]
        v = float(values[i])
        this_sign = np.sign(v)

        contiguous = (s == last_end)

        if contiguous and this_sign == last_sign:
            cur["end"] = e
            cur["value"] += v
            cur["tok_spans"].append((s, e))
            last_end = e
        else:
            out.append(cur)
            cur = {
                "start": s,
                "end": e,
                "text": None,
                "value": v,
                "tok_spans": [(s, e)],
            }
            last_end = e
            last_sign = this_sign

    out.append(cur)
    return out


def _spans_from_values(
    text: str,
    values: np.ndarray,
    tokenizer,
    target_idx: int,
    k: int,
    max_length: int,
    merge: bool,
) -> List[Dict[str, Any]]:
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    offsets = enc["offset_mapping"]
    tok_ids = enc["input_ids"]

    n = min(len(offsets), len(values))
    values = np.array(values[:n], dtype=float)
    offsets = offsets[:n]

    if merge:
        merged = merge_adjacent_tokens(tok_ids[:n], offsets, values)

        for m in merged:
            m["text"] = text[m["start"]:m["end"]]

        mags = np.array([abs(m["value"]) for m in merged])
        nz = normalize_abs(mags)

        for i, m in enumerate(merged):
            m["score"] = float(nz[i])
            m["sign"] = int(np.sign(m["value"])) if m["value"] != 0 else 0
            m["source"] = "shap"

        merged.sort(key=lambda d: abs(d["value"]), reverse=True)
        top = merged[:k]

    else:
        mags = np.abs(values)
        nz = normalize_abs(mags)

        spans = []
        for (s, e), v, sc in zip(offsets, values, nz):
            if s == e:
                continue
            spans.append({
                "start": int(s),
                "end": int(e),
                "text": text[s:e],
                "value": float(v),
                "score": float(sc),
                "sign": int(np.sign(v)) if v != 0 else 0,
                "source": "shap",
            })

        spans.sort(key=lambda d: abs(d["value"]), reverse=True)
        top = spans[:k]

    out = []
    for d in top:
        out.append({
            "start": int(d["start"]),
            "end": int(d["end"]),
            "text": str(d["text"]),
            "score": float(d["score"]),
            "value": float(d["value"]),
            "sign": int(d["sign"]),
            "source": "shap",
        })

    return out


class ShapSpanExplainer:
    def __init__(self, tokenizer, model, device: str = "cpu", max_length: int = 512):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.max_length = max_length

        masker = shap.maskers.Text(tokenizer)
        self.explainer = shap.Explainer(
            lambda s: predict_proba(list(s), tokenizer, model, device, max_length),
            masker=masker,
            algorithm="partition",
        )

    def explain_one(
        self,
        text: str,
        target_idx: int,
        k: int = 6,
        merge: bool = True,
    ) -> List[Dict[str, Any]]:
        if not text or not text.strip():
            return []

        sv = self.explainer([text])
        values = sv.values[0][:, target_idx]

        return _spans_from_values(
            text,
            values,
            self.tokenizer,
            target_idx,
            k,
            self.max_length,
            merge,
        )

    def explain_batch(
        self,
        texts: List[str],
        target_idx: int,
        k: int = 6,
        merge: bool = True,
    ) -> List[List[Dict[str, Any]]]:
        clean = []
        index_map = []
        for i, t in enumerate(texts):
            if t and t.strip():
                clean.append(t)
                index_map.append(i)

        if not clean:
            return [[] for _ in texts]

        sv = self.explainer(clean)
        results = [[] for _ in texts]

        for pos, (text, sv_single) in enumerate(zip(clean, sv)):
            values = sv_single.values[:, target_idx]
            spans = _spans_from_values(
                text,
                values,
                self.tokenizer,
                target_idx,
                k,
                self.max_length,
                merge,
            )
            results[index_map[pos]] = spans

        return results

def explain_with_shap_spans(
    text: str,
    tokenizer,
    model,
    device: str,
    target_idx: int,
    k: int = 6,
    max_length: int = 512,
    merge: bool = True,
) -> List[Dict[str, Any]]:
    expl = ShapSpanExplainer(tokenizer, model, device=device, max_length=max_length)
    return expl.explain_one(text, target_idx=target_idx, k=k, merge=merge)
