from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure project root is on sys.path so `src` imports work from any cwd.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data import ItemType, load_interactions, load_items
from src.models import build_collab_model, build_content_model
from src.recommender import (
    recommend_for_user,
    recommend_from_history,
    recommend_from_text,
    recommend_popular,
    recommend_similar_items,
)

app = FastAPI(title="Recommendation Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ItemRequest(BaseModel):
    item_id: str
    k: int = 5
    item_type: Optional[ItemType] = "all"
    diversity_lambda: float = 1.0
    mix_types: bool = False


class UserRequest(BaseModel):
    user_id: str
    k: int = 5
    item_type: Optional[ItemType] = "all"
    alpha: float = 0.6
    recency_half_life_days: Optional[float] = None
    diversity_lambda: float = 1.0
    mix_types: bool = False


class HistoryRequest(BaseModel):
    items: list[str]
    weights: Optional[list[float]] = None
    k: int = 5
    item_type: Optional[ItemType] = "all"
    alpha: float = 0.6
    diversity_lambda: float = 1.0
    mix_types: bool = False


class TextRequest(BaseModel):
    text: str
    k: int = 5
    item_type: Optional[ItemType] = "all"
    diversity_lambda: float = 1.0
    mix_types: bool = False


class PopularRequest(BaseModel):
    k: int = 5
    item_type: Optional[ItemType] = "all"


class ItemsResponse(BaseModel):
    items: list[dict]


class UsersResponse(BaseModel):
    users: list[str]


def _load_state():
    items_df = load_items("all")
    interactions_df = load_interactions()
    vectorizer, tfidf_matrix, content_sim = build_content_model(items_df)
    collab_sim = build_collab_model(interactions_df, items_df)
    return items_df, interactions_df, vectorizer, tfidf_matrix, content_sim, collab_sim


STATE = _load_state()


def _serialize(df):
    if df is None or df.empty:
        return []
    return df.to_dict(orient="records")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/items", response_model=ItemsResponse)
def get_items(item_type: Optional[ItemType] = "all"):
    items_df = STATE[0]
    if item_type and item_type != "all":
        items_df = items_df[items_df["type"] == item_type]
    return {"items": _serialize(items_df)}


@app.get("/users", response_model=UsersResponse)
def get_users():
    interactions_df = STATE[1]
    users = sorted(interactions_df["user_id"].unique().tolist())
    return {"users": users}


@app.post("/recommend/item", response_model=ItemsResponse)
def recommend_item(req: ItemRequest):
    items_df, _, _, _, content_sim, _ = STATE
    df = recommend_similar_items(
        item_id=req.item_id,
        items_df=items_df,
        sim_matrix=content_sim,
        k=req.k,
        item_type=req.item_type,
        diversity_lambda=req.diversity_lambda,
        mix_types=req.mix_types,
    )
    return {"items": _serialize(df)}


@app.post("/recommend/user", response_model=ItemsResponse)
def recommend_user(req: UserRequest):
    items_df, interactions_df, _, _, content_sim, collab_sim = STATE
    df = recommend_for_user(
        user_id=req.user_id,
        interactions_df=interactions_df,
        items_df=items_df,
        content_sim=content_sim,
        collab_sim=collab_sim,
        alpha=req.alpha,
        k=req.k,
        item_type=req.item_type,
        recency_half_life_days=req.recency_half_life_days,
        diversity_lambda=req.diversity_lambda,
        mix_types=req.mix_types,
    )
    return {"items": _serialize(df)}


@app.post("/recommend/history", response_model=ItemsResponse)
def recommend_history(req: HistoryRequest):
    items_df, interactions_df, _, _, content_sim, collab_sim = STATE
    df = recommend_from_history(
        history_items=req.items,
        history_weights=req.weights,
        interactions_df=interactions_df,
        items_df=items_df,
        content_sim=content_sim,
        collab_sim=collab_sim,
        alpha=req.alpha,
        k=req.k,
        item_type=req.item_type,
        diversity_lambda=req.diversity_lambda,
        mix_types=req.mix_types,
    )
    return {"items": _serialize(df)}


@app.post("/recommend/text", response_model=ItemsResponse)
def recommend_text(req: TextRequest):
    items_df, _, vectorizer, tfidf_matrix, content_sim, _ = STATE
    df = recommend_from_text(
        query=req.text,
        items_df=items_df,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        content_sim=content_sim,
        k=req.k,
        item_type=req.item_type,
        diversity_lambda=req.diversity_lambda,
        mix_types=req.mix_types,
    )
    return {"items": _serialize(df)}


@app.post("/recommend/popular", response_model=ItemsResponse)
def recommend_popular_api(req: PopularRequest):
    items_df, interactions_df, _, _, _, _ = STATE
    df = recommend_popular(
        items_df=items_df,
        interactions_df=interactions_df,
        k=req.k,
        item_type=req.item_type,
    )
    return {"items": _serialize(df)}
