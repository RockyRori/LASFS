# LASFS Web API

FastAPI backend for the CSV upload demo.

## Run

```bash
pip install -r requirements.txt
python run_server.py
```

The API exposes:

- `GET /health`
- `POST /api/preview`
- `POST /api/analyze`

`POST /api/analyze` accepts an optional `deepseek_api_key` form field.
When it is present, semantic leakage scores are generated through DeepSeek.
When it is empty, the API uses the offline heuristic scorer.
