from __future__ import annotations

from io import BytesIO

import pandas as pd
from fastapi import HTTPException, UploadFile


MAX_UPLOAD_BYTES = 25 * 1024 * 1024


async def read_uploaded_csv(file: UploadFile) -> pd.DataFrame:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="CSV file is larger than 25 MB.")

    try:
        frame = pd.read_csv(BytesIO(content))
    except UnicodeDecodeError:
        try:
            frame = pd.read_csv(BytesIO(content), encoding="gbk")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Could not decode CSV: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc

    if frame.empty:
        raise HTTPException(status_code=400, detail="CSV has no rows.")
    if len(frame.columns) < 2:
        raise HTTPException(status_code=400, detail="CSV must contain at least one feature and one target column.")
    return frame
