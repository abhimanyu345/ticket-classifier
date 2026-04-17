import time
import torch
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram
import numpy as np

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "models/distilbert-ticket-classifier"
MAX_LENGTH = 128

# ── Custom Prometheus metrics ─────────────────────────────────────────────────
PREDICTION_COUNTER = Counter(
    "ticket_predictions_total",
    "Total number of ticket predictions",
    ["predicted_class"]
)
PREDICTION_LATENCY = Histogram(
    "ticket_prediction_latency_seconds",
    "Time spent on each prediction"
)
CONFIDENCE_HISTOGRAM = Histogram(
    "ticket_prediction_confidence",
    "Confidence score of predictions",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
)

# ── Global model state ────────────────────────────────────────────────────────
model_state = {}

# ── Startup / shutdown ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model_state["tokenizer"] = tokenizer
    model_state["model"]     = model
    model_state["device"]    = device
    model_state["loaded_at"] = time.time()

    logger.info(f"Model loaded on {device}")
    yield

    model_state.clear()
    logger.info("Model unloaded")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Ticket Classifier API",
    description="Classifies customer support tickets using DistilBERT",
    version="1.0.0",
    lifespan=lifespan
)

# Attach Prometheus instrumentation
Instrumentator().instrument(app).expose(app)

# ── Schemas ───────────────────────────────────────────────────────────────────
class TicketRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "I was charged twice for my subscription this month"
            }
        }

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence:      float
    all_scores:      dict[str, float]
    latency_ms:      float

class HealthResponse(BaseModel):
    status:      str
    model:       str
    device:      str
    uptime_secs: float

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    if "model" not in model_state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status":      "healthy",
        "model":       MODEL_PATH,
        "device":      model_state["device"],
        "uptime_secs": round(time.time() - model_state["loaded_at"], 2)
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: TicketRequest):
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="Text cannot be empty")

    tokenizer = model_state["tokenizer"]
    model     = model_state["model"]
    device    = model_state["device"]

    start = time.time()

    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
    pred_idx = int(np.argmax(probs))
    labels   = model.config.id2label

    predicted_class = labels[pred_idx]
    confidence      = float(probs[pred_idx])
    all_scores      = {labels[i]: round(float(p), 4) for i, p in enumerate(probs)}
    latency_ms      = round((time.time() - start) * 1000, 2)

    # Update Prometheus metrics
    PREDICTION_COUNTER.labels(predicted_class=predicted_class).inc()
    PREDICTION_LATENCY.observe(latency_ms / 1000)
    CONFIDENCE_HISTOGRAM.observe(confidence)

    logger.info(f"Predicted: {predicted_class} ({confidence:.2%}) in {latency_ms}ms")

    return {
        "predicted_class": predicted_class,
        "confidence":      round(confidence, 4),
        "all_scores":      all_scores,
        "latency_ms":      latency_ms
    }

@app.get("/")
def root():
    return {
        "message": "Ticket Classifier API",
        "docs":    "/docs",
        "health":  "/health",
        "predict": "/predict",
        "metrics": "/metrics"
    }