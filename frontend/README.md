# LASFS Frontend

Vite + React + TypeScript interface for uploading a CSV file and comparing
Random Forest feature selection with LASFS.

## Run

```bash
npm install
npm run dev
```

The frontend expects the FastAPI backend at `http://127.0.0.1:8000`.
Override it with `VITE_API_BASE` if needed.

The DeepSeek API key field is optional. Empty input uses offline scoring; a
provided key is sent only with the current analysis request.
