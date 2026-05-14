from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

ItemType = Literal["products", "music", "all"]


def load_products() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "products.csv")
    df["type"] = "products"
    return df


def load_music() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "music.csv")
    df["type"] = "music"
    return df


def load_interactions() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "interactions.csv")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def load_items(item_type: ItemType = "all") -> pd.DataFrame:
    if item_type == "products":
        return load_products()
    if item_type == "music":
        return load_music()

    products = load_products()
    music = load_music()
    return pd.concat([products, music], ignore_index=True)


def build_text_field(df: pd.DataFrame) -> pd.Series:
    # Combine any relevant columns into a single text field for TF-IDF.
    cols = [
        "title",
        "brand",
        "category",
        "tags",
        "artist",
        "genre",
    ]
    parts = []
    for col in cols:
        if col in df.columns:
            parts.append(df[col].fillna(""))

    if not parts:
        return pd.Series(["" for _ in range(len(df))])

    text = parts[0].astype(str)
    for part in parts[1:]:
        text = text + " " + part.astype(str)
    return text.str.lower()


def filter_items(df: pd.DataFrame, item_type: Optional[ItemType]) -> pd.DataFrame:
    if item_type is None or item_type == "all":
        return df
    return df[df["type"] == item_type].copy()
