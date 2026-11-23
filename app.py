"""FastAPI inference service for tabular phishing-website features."""
import hmac
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

load_dotenv()
app = FastAPI(title="NetworkSecurity MLOps", description="CSV-based phishing website classification")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])
templates = Jinja2Templates(directory="templates")

@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/train", tags=["training"])
async def train_route(x_api_key: str | None = Header(default=None)):
    expected = os.getenv("TRAINING_API_KEY")
    if not expected:
        raise HTTPException(status_code=403, detail="API training disabled; run python main.py locally")
    if not hmac.compare_digest(x_api_key or "", expected):
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        TrainingPipeline().run_pipeline()
        return {"status": "training completed"}
    except Exception as e:
        raise NetworkSecurityException(e, sys)

@app.post("/predict", tags=["prediction"])
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        data = pd.read_csv(file.file)
        preprocessor = load_object("final_model/preprocessor.pkl")
        model = load_object("final_model/model.pkl")
        predictor = NetworkModel(preprocessor=preprocessor, model=model)
        predictions = predictor.predict(data)
        data["predicted_column"] = predictions
        os.makedirs("prediction_output", exist_ok=True)
        data.to_csv("prediction_output/output.csv", index=False)
        return templates.TemplateResponse(
            "table.html", {"request": request, "table": data.to_html(classes="table table-striped", index=False)}
        )
    except Exception as e:
        raise NetworkSecurityException(e, sys)
