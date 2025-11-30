from pydantic import BaseModel
from typing import List, Optional, Dict

# what the user sends to /predict
class PredictRequest(BaseModel):
    title: Optional[str] = None
    text: str
    useShap: Optional[bool] = False

# model’s prediction about bias
class BiasOut(BaseModel):
    label: str
    confidence: float
    probs: Optional[Dict[str, float]] = None

# span/keywords to show the user
class RationaleSpan(BaseModel):
    start: int
    end: int
    text: str
    score: Optional[float] = None     
    value: Optional[float] = None      
    sign: Optional[int] = None         

# container for all spans to highlight
class ExplainOut(BaseModel):
    spans: List[RationaleSpan] = []

# what the API returns for /predict
class PredictResponse(BaseModel):
    summary: str
    bias: BiasOut
    explain: ExplainOut
    source_prior: Optional[Dict[str, str]] = None   # e.g., {"source":"cnn.com","rating":"Left","origin":"AllSides"}

# basic runtime info for debugging
class EnvResponse(BaseModel):
    device: str
    torch: str
    cuda_available: bool
    cuda_version: Optional[str] = None
