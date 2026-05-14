from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data import build_text_field


def build_content_model(
    items_df: pd.DataFrame,
) -> Tuple[TfidfVectorizer, np.ndarray, np.ndarray]:
    text = build_text_field(items_df)
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(text)
    sim = cosine_similarity(matrix)
    return vectorizer, matrix, sim


def build_collab_model(
    interactions_df: pd.DataFrame, items_df: pd.DataFrame
) -> np.ndarray:
    # Build user-item matrix with all items as columns.
    item_ids = items_df["item_id"].tolist()
    matrix = (
        interactions_df.pivot_table(
            index="user_id", columns="item_id", values="interaction", fill_value=0
        )
        .reindex(columns=item_ids, fill_value=0)
    )

    # Item-item similarity based on user interaction vectors.
    sim = cosine_similarity(matrix.T)
    return sim
