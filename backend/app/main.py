from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import AnalyzeResponse, DatasetPreview
from app.services.csv_io import read_uploaded_csv
from app.services.feature_selection import analyze_features


app = FastAPI(title="LASFS Web API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/preview", response_model=DatasetPreview)
async def preview_dataset(file: UploadFile = File(...)) -> DatasetPreview:
    frame = await read_uploaded_csv(file)
    return DatasetPreview(
        filename=file.filename or "uploaded.csv",
        rows=len(frame),
        columns=[str(column) for column in frame.columns],
        preview=frame.head(8).where(frame.notna(), None).to_dict(orient="records"),
    )


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_dataset(
    file: UploadFile = File(...),
    target: str = Form(...),
    k_ratio: float = Form(0.4),
    deepseek_api_key: str = Form(""),
) -> AnalyzeResponse:
    frame = await read_uploaded_csv(file)
    try:
        return analyze_features(
            frame,
            file.filename or "uploaded.csv",
            target,
            k_ratio,
            deepseek_api_key=deepseek_api_key.strip() or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
