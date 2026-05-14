# Recommendation Engine (Products + Music)

This project now includes:
- **FastAPI backend** for recommendations
- **React + Vite frontend** for a modern UI
- **Recommendation quality controls** (recency bias, diversity, mixed-type reranking)

## Backend (FastAPI)

1) Create a virtual environment and install deps:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

2) Run the API:

```bash
uvicorn backend.main:app --reload --port 8000
```

API endpoints:
- `GET /health`
- `GET /items?item_type=all|products|music`
- `GET /users`
- `POST /recommend/item`
- `POST /recommend/user` (supports `recency_half_life_days`, `diversity_lambda`, `mix_types`)
- `POST /recommend/history` (supports `diversity_lambda`, `mix_types`)
- `POST /recommend/text` (supports `diversity_lambda`, `mix_types`)
- `POST /recommend/popular`

## Frontend (React + Vite)

1) Install deps:

```bash
cd frontend
npm install
```

2) Run dev server:

```bash
npm run dev
```

Then open the URL printed by Vite (usually `http://localhost:5173`).

If you change the API port, update `API_BASE` in:
- `frontend/src/App.jsx`

## Recommendation Quality Controls

- **Recency bias**: newer interactions count more using a half-life in days.
- **Diversity**: MMR-style reranking to avoid overly similar results.
- **Mix types**: when enabled and item_type is `all`, results alternate between products and music.

## Data

Sample datasets live in `data/`:
- `products.csv`
- `music.csv`
- `interactions.csv` (now includes a `timestamp` column)

You can replace them with your own data, keeping the same columns.

## Project Layout

```
C:\recomendation engine
+- backend
+- frontend
+- data
+- src
+- tests
```
