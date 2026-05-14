from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .data import ItemType, filter_items


def _minmax_scale(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    min_val = scores.min()
    max_val = scores.max()
    if max_val == min_val:
        return np.zeros_like(scores)
    return (scores - min_val) / (max_val - min_val)


def _build_item_index(items_df: pd.DataFrame) -> dict[str, int]:
    return {item_id: idx for idx, item_id in enumerate(items_df["item_id"].tolist())}


def _filter_indices(
    items_df: pd.DataFrame, item_type: Optional[ItemType]
) -> np.ndarray:
    if item_type is None or item_type == "all":
        return np.arange(len(items_df))
    return items_df.index[items_df["type"] == item_type].to_numpy()


def _infer_single_type(items_df: pd.DataFrame, item_ids: list[str]) -> Optional[ItemType]:
    if not item_ids:
        return None
    subset = items_df[items_df["item_id"].isin(item_ids)]
    if subset.empty:
        return None
    types = subset["type"].dropna().unique().tolist()
    if len(types) == 1:
        return types[0]
    return None


def _compute_recency_weights(
    interactions_df: pd.DataFrame, half_life_days: Optional[float]
) -> pd.Series:
    if half_life_days is None or half_life_days <= 0:
        return pd.Series([1.0] * len(interactions_df), index=interactions_df.index)
    if "timestamp" not in interactions_df.columns:
        return pd.Series([1.0] * len(interactions_df), index=interactions_df.index)

    now = interactions_df["timestamp"].max()
    if pd.isna(now):
        return pd.Series([1.0] * len(interactions_df), index=interactions_df.index)

    age_days = (now - interactions_df["timestamp"]).dt.days.fillna(0)
    decay = 0.5 ** (age_days / half_life_days)
    return decay


def _rerank_diverse(
    scores: np.ndarray,
    sim_matrix: np.ndarray,
    candidate_indices: np.ndarray,
    k: int,
    diversity_lambda: float,
) -> list[int]:
    if diversity_lambda >= 1.0:
        ranked = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]
        return ranked[:k].tolist()

    selected: list[int] = []
    remaining = set(candidate_indices.tolist())

    while remaining and len(selected) < k:
        best_idx = None
        best_score = -np.inf
        for idx in list(remaining):
            score = scores[idx]
            if not selected:
                mmr = score
            else:
                max_sim = max(sim_matrix[idx][s] for s in selected)
                mmr = diversity_lambda * score - (1 - diversity_lambda) * max_sim
            if mmr > best_score:
                best_score = mmr
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


def _rerank_mixed(
    items_df: pd.DataFrame, ranked_indices: list[int], k: int
) -> list[int]:
    if not ranked_indices:
        return []

    products = [i for i in ranked_indices if items_df.iloc[i]["type"] == "products"]
    music = [i for i in ranked_indices if items_df.iloc[i]["type"] == "music"]

    if not products or not music:
        return ranked_indices[:k]

    mixed: list[int] = []
    # Start with the type that has the higher top score (already ranked).
    start_with_products = ranked_indices[0] in products

    while (products or music) and len(mixed) < k:
        if start_with_products:
            if products:
                mixed.append(products.pop(0))
            if music and len(mixed) < k:
                mixed.append(music.pop(0))
        else:
            if music:
                mixed.append(music.pop(0))
            if products and len(mixed) < k:
                mixed.append(products.pop(0))

        if not products:
            mixed.extend(music[: max(0, k - len(mixed))])
            break
        if not music:
            mixed.extend(products[: max(0, k - len(mixed))])
            break

    return mixed[:k]


def recommend_similar_items(
    item_id: str,
    items_df: pd.DataFrame,
    sim_matrix: np.ndarray,
    k: int = 5,
    item_type: Optional[ItemType] = None,
    diversity_lambda: float = 1.0,
    mix_types: bool = False,
) -> pd.DataFrame:
    item_index = _build_item_index(items_df)
    if item_id not in item_index:
        return pd.DataFrame()

    if (item_type is None or item_type == "all") and not mix_types:
        inferred = _infer_single_type(items_df, [item_id])
        if inferred is not None:
            item_type = inferred

    idx = item_index[item_id]
    scores = sim_matrix[idx].copy()
    scores[idx] = -1  # exclude self

    allowed = _filter_indices(items_df, item_type)
    ranked = _rerank_diverse(scores, sim_matrix, allowed, k, diversity_lambda)
    if mix_types and (item_type is None or item_type == "all"):
        ranked = _rerank_mixed(items_df, ranked, k)

    result = items_df.iloc[ranked].copy()
    result["score"] = scores[ranked]
    return result.reset_index(drop=True)


def recommend_for_user(
    user_id: str,
    interactions_df: pd.DataFrame,
    items_df: pd.DataFrame,
    content_sim: np.ndarray,
    collab_sim: np.ndarray,
    alpha: float = 0.6,
    k: int = 5,
    item_type: Optional[ItemType] = None,
    recency_half_life_days: Optional[float] = None,
    diversity_lambda: float = 1.0,
    mix_types: bool = False,
) -> pd.DataFrame:
    item_index = _build_item_index(items_df)
    user_rows = interactions_df[interactions_df["user_id"] == user_id]

    if user_rows.empty:
        return recommend_popular(items_df, interactions_df, k=k, item_type=item_type)

    if (item_type is None or item_type == "all") and not mix_types:
        inferred = _infer_single_type(items_df, user_rows["item_id"].tolist())
        if inferred is not None:
            item_type = inferred

    # Build score vectors for content and collaborative components.
    scores_content = np.zeros(len(items_df))
    scores_collab = np.zeros(len(items_df))
    recency_weights = _compute_recency_weights(user_rows, recency_half_life_days)

    for (_, row), recency_w in zip(user_rows.iterrows(), recency_weights):
        item_id = row["item_id"]
        if item_id not in item_index:
            continue
        weight = float(row["interaction"]) * float(recency_w)
        idx = item_index[item_id]
        scores_content += content_sim[idx] * weight
        scores_collab += collab_sim[idx] * weight

    scores_content = _minmax_scale(scores_content)
    scores_collab = _minmax_scale(scores_collab)
    scores = alpha * scores_content + (1 - alpha) * scores_collab

    # Exclude items already interacted with.
    for item_id in user_rows["item_id"].tolist():
        if item_id in item_index:
            scores[item_index[item_id]] = -1

    allowed = _filter_indices(items_df, item_type)
    ranked = _rerank_diverse(scores, content_sim, allowed, k, diversity_lambda)
    if mix_types and (item_type is None or item_type == "all"):
        ranked = _rerank_mixed(items_df, ranked, k)

    result = items_df.iloc[ranked].copy()
    result["score"] = scores[ranked]
    return result.reset_index(drop=True)


def recommend_from_text(
    query: str,
    items_df: pd.DataFrame,
    vectorizer,
    tfidf_matrix,
    content_sim: np.ndarray,
    k: int = 5,
    item_type: Optional[ItemType] = None,
    diversity_lambda: float = 1.0,
    mix_types: bool = False,
) -> pd.DataFrame:
    if not query.strip():
        return pd.DataFrame()

    q_vec = vectorizer.transform([query.lower()])
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()

    allowed = _filter_indices(items_df, item_type)
    ranked = _rerank_diverse(sims, content_sim, allowed, k, diversity_lambda)
    if mix_types and (item_type is None or item_type == "all"):
        ranked = _rerank_mixed(items_df, ranked, k)

    result = items_df.iloc[ranked].copy()
    result["score"] = sims[ranked]
    return result.reset_index(drop=True)


def recommend_from_history(
    history_items: list[str],
    history_weights: Optional[list[float]],
    interactions_df: pd.DataFrame,
    items_df: pd.DataFrame,
    content_sim: np.ndarray,
    collab_sim: np.ndarray,
    alpha: float = 0.6,
    k: int = 5,
    item_type: Optional[ItemType] = None,
    diversity_lambda: float = 1.0,
    mix_types: bool = False,
) -> pd.DataFrame:
    if not history_items:
        return recommend_popular(items_df, interactions_df, k=k, item_type=item_type)

    if (item_type is None or item_type == "all") and not mix_types:
        inferred = _infer_single_type(items_df, history_items)
        if inferred is not None:
            item_type = inferred

    item_index = _build_item_index(items_df)
    scores_content = np.zeros(len(items_df))
    scores_collab = np.zeros(len(items_df))

    if history_weights and len(history_weights) != len(history_items):
        history_weights = None

    for i, item_id in enumerate(history_items):
        if item_id not in item_index:
            continue
        weight = float(history_weights[i]) if history_weights is not None else 1.0
        idx = item_index[item_id]
        scores_content += content_sim[idx] * weight
        scores_collab += collab_sim[idx] * weight

    scores_content = _minmax_scale(scores_content)
    scores_collab = _minmax_scale(scores_collab)
    scores = alpha * scores_content + (1 - alpha) * scores_collab

    for item_id in history_items:
        if item_id in item_index:
            scores[item_index[item_id]] = -1

    allowed = _filter_indices(items_df, item_type)
    ranked = _rerank_diverse(scores, content_sim, allowed, k, diversity_lambda)
    if mix_types and (item_type is None or item_type == "all"):
        ranked = _rerank_mixed(items_df, ranked, k)

    result = items_df.iloc[ranked].copy()
    result["score"] = scores[ranked]
    return result.reset_index(drop=True)


def recommend_popular(
    items_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    k: int = 5,
    item_type: Optional[ItemType] = None,
) -> pd.DataFrame:
    if interactions_df.empty:
        return items_df.head(k).copy().reset_index(drop=True)

    counts = (
        interactions_df.groupby("item_id")["interaction"].mean().rename("score")
    )
    merged = items_df.merge(counts, on="item_id", how="left").fillna({"score": 0})

    if item_type is not None and item_type != "all":
        merged = merged[merged["type"] == item_type]

    result = merged.sort_values("score", ascending=False).head(k)
    return result.reset_index(drop=True)
