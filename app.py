from __future__ import annotations

import streamlit as st

from src.data import ItemType, load_interactions, load_items
from src.models import build_collab_model, build_content_model
from src.recommender import (
    recommend_for_user,
    recommend_from_text,
    recommend_popular,
    recommend_similar_items,
)

st.set_page_config(page_title="Recommendation Engine", page_icon="??", layout="wide")

st.title("Recommendation Engine: Products + Music")
st.caption("Hybrid recommendations using content and collaborative signals")


@st.cache_data
def get_items():
    return load_items("all")


@st.cache_data
def get_interactions():
    return load_interactions()


@st.cache_resource
def get_models(items_df, interactions_df):
    vectorizer, tfidf_matrix, content_sim = build_content_model(items_df)
    collab_sim = build_collab_model(interactions_df, items_df)
    return vectorizer, tfidf_matrix, content_sim, collab_sim


items_df = get_items()
interactions_df = get_interactions()
vectorizer, tfidf_matrix, content_sim, collab_sim = get_models(
    items_df, interactions_df
)

with st.sidebar:
    st.subheader("Filters")
    item_type: ItemType = st.selectbox(
        "Item Type", options=["all", "products", "music"], index=0
    )
    k = st.slider("How Many Results", min_value=3, max_value=10, value=5)

    st.subheader("Hybrid Weight")
    alpha = st.slider(
        "Content vs Collaborative (alpha)", min_value=0.0, max_value=1.0, value=0.6
    )

st.divider()

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Try It")
    tabs = st.tabs(["By Item", "By User", "By Text", "Popular"])

    with tabs[0]:
        item_id = st.selectbox(
            "Choose an Item", options=items_df["item_id"].tolist()
        )
        if st.button("Recommend Similar Items", use_container_width=True):
            recs = recommend_similar_items(
                item_id=item_id,
                items_df=items_df,
                sim_matrix=content_sim,
                k=k,
                item_type=item_type,
            )
            st.dataframe(recs, use_container_width=True)

    with tabs[1]:
        user_id = st.selectbox(
            "Choose a User", options=interactions_df["user_id"].unique().tolist()
        )
        if st.button("Recommend for User", use_container_width=True):
            recs = recommend_for_user(
                user_id=user_id,
                interactions_df=interactions_df,
                items_df=items_df,
                content_sim=content_sim,
                collab_sim=collab_sim,
                alpha=alpha,
                k=k,
                item_type=item_type,
            )
            st.dataframe(recs, use_container_width=True)

    with tabs[2]:
        query = st.text_input("Describe what you want")
        if st.button("Recommend from Text", use_container_width=True):
            recs = recommend_from_text(
                query=query,
                items_df=items_df,
                vectorizer=vectorizer,
                tfidf_matrix=tfidf_matrix,
                k=k,
                item_type=item_type,
            )
            st.dataframe(recs, use_container_width=True)

    with tabs[3]:
        if st.button("Show Popular", use_container_width=True):
            recs = recommend_popular(
                items_df=items_df,
                interactions_df=interactions_df,
                k=k,
                item_type=item_type,
            )
            st.dataframe(recs, use_container_width=True)

with col2:
    st.subheader("Data Preview")
    st.write("Items")
    st.dataframe(items_df.head(8), use_container_width=True)
    st.write("Interactions")
    st.dataframe(interactions_df.head(8), use_container_width=True)
