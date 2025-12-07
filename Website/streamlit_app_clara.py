# streamlit_app_filmlytics.py

import os
import json
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pymongo import MongoClient
from pymongo.server_api import ServerApi
import certifi
import joblib  # For loading models / scalers

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

# MongoDB Connection (from Streamlit secrets)
MONGODB_URI = st.secrets["MONGODB_URI"]

st.set_page_config(
    page_title="Filmlytics – Cinemaniacs",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global CSS for modern / professional styling
st.markdown(
    """
    <style>
    /* Make main page wider */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    /* Section titles */
    h1, h2, h3 {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    /* Card-like expander */
    .streamlit-expanderHeader {
        font-weight: 600;
    }
    /* Subtle divider spacing */
    hr {
        margin: 1.5rem 0;
    }
    /* Metric text tweaks */
    .stMetric {
        text-align: center;
    }
    /* Reduce default emoji size if present in headers */
    h1 span, h2 span, h3 span {
        font-size: 1.1em !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# DATABASE CONNECTION
# -----------------------------------------------------------------------------

@st.cache_resource
def get_database_connection():
    """Connect to MongoDB and cache the connection."""
    try:
        client = MongoClient(
            MONGODB_URI,
            server_api=ServerApi("1"),
            tlsCAFile=certifi.where(),
        )
        db = client["cinemaniacs"]
        # Test connection
        db.movies.count_documents({})
        return db
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None


# -----------------------------------------------------------------------------
# ARTIFACT LOADING AND ENSEMBLE PREDICTION
# -----------------------------------------------------------------------------

@st.cache_resource
def load_ensemble_artifacts():
    """
    Load all models, scalers, prediction dataframes, and stacking metadata.

    This is the same logic as the working version, with:
    - GNN / KGCN / XGB precomputed predictions from CSV
    - Optional stacking_meta_model.pkl
    - ensemble_weights.json or default weights
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    artifact_dir = os.path.join(script_dir, ".", "model_artifacts")

    artifacts = {}

    required_files = {
        # XGBoost
        "xg_model": "xgboost_base_model.pkl",
        "xg_features": "xg_feature_columns.json",
        # GNN/KGCN shared
        "scaler": "movie_feature_scaler_diversity.pkl",
        # GNN Predictions (Lookup)
        "gnn_preds": "gnn_preds_all_movies.csv",
        # KGCN Predictions (Lookup)
        "kgcn_preds": "kgcn_preds_all_movies.csv",
        # XGBoost Predictions (Lookup)
        "xgb_preds": "xgb_preds_all_movies.csv",
    }

    for key, filename in required_files.items():
        path = os.path.join(artifact_dir, filename)
        if not os.path.exists(path):
            st.warning(f"Missing required file: {filename}. Prediction will be incomplete.")
            artifacts[key] = None
            continue

        try:
            if filename.endswith(".pkl"):
                artifacts[key] = joblib.load(path)
            elif filename.endswith(".json"):
                with open(path, "r") as f:
                    artifacts[key] = json.load(f)
            elif filename.endswith(".csv"):
                df = pd.read_csv(path)
                df["tmdb_id"] = (
                    pd.to_numeric(df["tmdb_id"], errors="coerce")
                    .fillna(0)
                    .astype(int)
                )

                if "pred_audience_score" in df.columns:
                    artifacts[key] = df.set_index("tmdb_id")["pred_audience_score"]
                elif "predicted_audience_score" in df.columns:
                    artifacts[key] = df.set_index("tmdb_id")["predicted_audience_score"]
                else:
                    artifacts[key] = df.set_index("tmdb_id").iloc[:, 0]
        except Exception as e:
            return None, f"Error loading {filename}: {e}"

    # Stacking meta-model
    stacking_path = os.path.join(artifact_dir, "stacking_meta_model.pkl")
    if os.path.exists(stacking_path):
        try:
            with open(stacking_path, "rb") as f:
                artifacts["stacking_model"] = pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load stacking model: {e}")
            artifacts["stacking_model"] = None
    else:
        artifacts["stacking_model"] = None

    # Ensemble metadata (e.g., RMSE, meta-model name)
    meta_path = os.path.join(artifact_dir, "ensemble_weights.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                artifacts["ensemble_meta"] = json.load(f)
        except Exception:
            artifacts["ensemble_meta"] = {}
    else:
        artifacts["ensemble_meta"] = {}

    # Fallback ensemble weights
    artifacts["ensemble_weights"] = {"gnn": 0.33, "kgcn": 0.34, "xg": 0.33}

    return artifacts, None


def safe_get_prediction(preds, tmdb_id):
    """Safely get a prediction value from a pandas Series."""
    if preds is None:
        return np.nan
    val = preds.get(tmdb_id, np.nan)
    if isinstance(val, pd.Series):
        val = val.iloc[0] if len(val) > 0 else np.nan
    if pd.isna(val):
        return np.nan
    return float(val)


def predict_ensemble(movie_data, artifacts):
    """
    Use stacking meta-learner to combine GNN, KGCN, and XGBoost predictions.
    Falls back to weighted average if stacking model is not available.

    Returns
    -------
    ensemble_prediction : float in [0, 1] or NaN
    prediction_breakdown : dict with keys 'gnn', 'kgcn', 'xg'
    """
    tmdb_id = movie_data.get("tmdb_id")
    if tmdb_id is None:
        return np.nan, {}

    gnn_pred = safe_get_prediction(artifacts.get("gnn_preds"), tmdb_id)
    kgcn_pred = safe_get_prediction(artifacts.get("kgcn_preds"), tmdb_id)
    xgb_pred = safe_get_prediction(artifacts.get("xgb_preds"), tmdb_id)

    predictions = {
        "gnn": gnn_pred,
        "kgcn": kgcn_pred,
        "xg": xgb_pred,
    }

    stacking_model = artifacts.get("stacking_model")
    has_all_preds = not (
        np.isnan(gnn_pred) or np.isnan(kgcn_pred) or np.isnan(xgb_pred)
    )

    # Use stacking model if available + all three predictions exist
    if stacking_model is not None and has_all_preds:
        X = np.array([[gnn_pred, kgcn_pred, xgb_pred]])
        ensemble_pred = float(np.clip(stacking_model.predict(X)[0], 0, 1))
        return ensemble_pred, predictions

    # Fallback: weighted average
    weights = artifacts.get(
        "ensemble_weights", {"gnn": 0.33, "kgcn": 0.34, "xg": 0.33}
    )
    valid_preds = {k: v for k, v in predictions.items() if not np.isnan(v)}

    if not valid_preds:
        tmdb_avg = movie_data.get("tmdb_metrics", {}).get("vote_average")
        if tmdb_avg is not None:
            return float(tmdb_avg) / 10.0, predictions
        return np.nan, predictions

    valid_keys = valid_preds.keys()
    total_valid_weight = sum(weights.get(k, 0) for k in valid_keys)

    if total_valid_weight == 0:
        ensemble_pred = np.mean(list(valid_preds.values()))
    else:
        ensemble_pred = sum(
            valid_preds[k] * (weights.get(k, 0) / total_valid_weight)
            for k in valid_keys
        )

    ensemble_pred = float(np.clip(ensemble_pred, 0.0, 1.0))
    return ensemble_pred, predictions


# -----------------------------------------------------------------------------
# DATA QUERY FUNCTIONS
# -----------------------------------------------------------------------------

@st.cache_data
def get_all_movie_titles(_db):
    """Get a list of UNIQUE movie titles, sorted by vote count (popularity)."""
    try:
        pipeline = [
            {
                "$match": {
                    "title": {"$ne": None},
                    "tmdb_id": {"$ne": None},
                    "tmdb_metrics.vote_count": {"$gte": 1},
                }
            },
            {
                "$group": {
                    "_id": "$tmdb_id",
                    "title": {"$first": "$title"},
                    "vote_count": {"$max": "$tmdb_metrics.vote_count"},
                }
            },
            {"$sort": {"vote_count": -1}},
            {"$project": {"_id": 0, "title": 1}},
        ]
        unique_titles = list(_db.movies.aggregate(pipeline))
        return [doc["title"] for doc in unique_titles]
    except Exception as e:
        st.error(f"Error fetching unique movie titles: {e}")
        return []


def search_movie(db, title):
    movie = db.movies.find_one(
        {"title": {"$regex": f"^{title}$", "$options": "i"}}
    )
    if not movie:
        movie = db.movies.find_one({"title": {"$regex": title, "$options": "i"}})
    return movie


def get_top_movies(db, limit=50, min_votes=1000):
    query = {
        "tmdb_metrics.vote_count": {"$gte": min_votes},
        "tmdb_metrics.vote_average": {"$ne": None},
    }
    return (
        list(
            db.movies.find(query)
            .sort("tmdb_metrics.vote_average", -1)
            .limit(limit)
        )
    )


def get_similar_movies(db, tmdb_id, limit=10, min_votes=1000):
    movie = db.movies.find_one({"tmdb_id": tmdb_id})
    if not movie:
        return []

    genres = movie["production"].get("genres", [])
    if not genres:
        return []

    genre_count = len(genres)

    pipeline = [
        {
            "$match": {
                "production.genres": {
                    "$all": genres,
                    "$size": genre_count,
                },
                "tmdb_id": {"$ne": tmdb_id},
                "tmdb_metrics.vote_count": {"$gte": min_votes},
                "tmdb_metrics.vote_average": {"$ne": None},
            }
        },
        {
            "$group": {
                "_id": "$tmdb_id",
                "unique_movie": {"$first": "$$ROOT"},
            }
        },
        {"$replaceRoot": {"newRoot": "$unique_movie"}},
        {"$sort": {"tmdb_metrics.vote_average": -1}},
        {"$limit": limit},
    ]
    return list(db.movies.aggregate(pipeline))


def get_database_stats(db):
    total = db.movies.count_documents({})
    successful = db.movies.count_documents({"tmdb_metrics.is_successful": True})
    with_rt = db.movies.count_documents(
        {"rotten_tomatoes.has_rt_url": True}
    )
    with_trailer = db.movies.count_documents(
        {"trailer.trailer_url_youtube": {"$ne": None}}
    )

    return {
        "total": total,
        "successful": successful,
        "with_rotten_tomatoes": with_rt,
        "with_trailers": with_trailer,
    }


def get_all_genres(db):
    genres = db.movies.distinct("production.genres")
    return sorted([g for g in genres if g])


def get_movies_by_genre(db, genre, limit=20, min_votes=1000):
    pipeline = [
        {
            "$match": {
                "production.genres": genre,
                "tmdb_metrics.vote_count": {"$gte": min_votes},
                "tmdb_metrics.vote_average": {"$ne": None},
            }
        },
        {
            "$group": {
                "_id": "$tmdb_id",
                "unique_movie": {"$first": "$$ROOT"},
            }
        },
        {"$replaceRoot": {"newRoot": "$unique_movie"}},
        {"$sort": {"tmdb_metrics.vote_average": -1}},
        {"$limit": limit},
    ]
    return list(db.movies.aggregate(pipeline))


# -----------------------------------------------------------------------------
# VISUALIZATION FUNCTIONS
# -----------------------------------------------------------------------------

def create_genre_distribution_chart(db):
    pipeline = [
        {"$unwind": "$production.genres"},
        {"$group": {"_id": "$production.genres", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 15},
    ]
    results = list(db.movies.aggregate(pipeline))
    df = pd.DataFrame(results)
    if df.empty:
        return None
    df.columns = ["Genre", "Count"]
    fig = px.bar(
        df,
        x="Genre",
        y="Count",
        title="Top 15 Movie Genres",
        color="Count",
        color_continuous_scale="viridis",
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def create_rating_distribution(db):
    movies = list(
        db.movies.find(
            {"tmdb_metrics.vote_average": {"$ne": None}},
            {"tmdb_metrics.vote_average": 1},
        ).limit(5000)
    )

    ratings = [m["tmdb_metrics"]["vote_average"] for m in movies]
    if not ratings:
        return None

    fig = px.histogram(
        ratings,
        nbins=50,
        title="Distribution of Movie Ratings (TMDB)",
        labels={"value": "Rating", "count": "Number of Movies"},
        color_discrete_sequence=["#1f77b4"],
    )
    fig.update_layout(showlegend=False)
    return fig


def create_success_over_time(db):
    pipeline = [
        {
            "$match": {
                "release_info.tmdb_release_date": {
                    "$ne": None,
                    "$regex": "^[0-9]{4}",
                }
            }
        },
        {
            "$project": {
                "year": {
                    "$substr": ["$release_info.tmdb_release_date", 0, 4]
                },
                "is_successful": "$tmdb_metrics.is_successful",
            }
        },
        {
            "$group": {
                "_id": "$year",
                "total": {"$sum": 1},
                "successful": {
                    "$sum": {"$cond": ["$is_successful", 1, 0]}
                },
            }
        },
        {
            "$project": {
                "year": "$_id",
                "success_rate": {
                    "$multiply": [
                        {"$divide": ["$successful", "$total"]},
                        100,
                    ]
                },
            }
        },
        {"$sort": {"year": 1}},
    ]

    results = list(db.movies.aggregate(pipeline))
    df = pd.DataFrame(results)

    if not df.empty and len(df) > 10:
        df = df[df["year"].astype(int) >= 2000]
        if df.empty:
            return None

        fig = px.line(
            df,
            x="year",
            y="success_rate",
            title="Movie Success Rate Over Time",
            labels={
                "year": "Year",
                "success_rate": "Success Rate (%)",
            },
            markers=True,
        )
        fig.update_layout(hovermode="x unified")
        return fig
    return None


# -----------------------------------------------------------------------------
# PAGE FUNCTIONS
# -----------------------------------------------------------------------------

def introduction_page():
    st.title("Filmlytics – Predicting Film Audience Scores with Graph-Based Models")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Project Overview")
        st.write(
            """
            Filmlytics is a predictive analytics platform that estimates Rotten Tomatoes audience
            scores for films using a hybrid ensemble of **Graph Neural Networks (GNN)**,
            **Knowledge-Graph Convolutional Networks (KGCN)**, and **XGBoost**.

            The system is built on a dataset of **66,000+ films (2010–2025)** that integrates:
            - TMDB metadata (genres, runtime, cast, production, release information)
            - Rotten Tomatoes audience and critic scores
            - YouTube trailer engagement and sentiment
            - Gender representation indicators in cast and crew
            """
        )
    with col2:
        st.markdown("**Course:** STA 160 – Data Science Project")
        st.markdown("**Team:** Cinemaniacs (Team 15)")
        st.markdown("**Focus:** Graph-based modeling & ensemble prediction")

    st.subheader("Motivation")
    st.write(
        """
        The film industry faces increasing pressure to predict audience reception while also 
        addressing debates about representation and diversity. Audience scores influence:
        - Marketing and advertising strategies  
        - Streaming and distribution decisions  
        - Long-term commercial success and franchise planning  

        However, systematic analysis of films beyond basic attributes (budget, runtime, genre)
        is constrained by **data fragmentation**:
        - Film metadata, engagement metrics, and diversity signals live across different platforms  
        - Gender and demographic information are not systematically tracked  
        - Simple regression and black-box models struggle to capture **rich relational structure**  
        
        By integrating multiple data sources, enriching them with **gender representation features**, 
        and building a **structured graphical model**, Filmlytics explores how production metadata, 
        engagement patterns, and relational similarity jointly shape audience response.
        """
    )

    st.subheader("Objectives")
    st.write(
        """
        - Construct a unified movie dataset from **TMDB**, **Rotten Tomatoes**, and the **YouTube API**  
        - Engineer feature-rich representations including sentiment and gender representation  
        - Build **GNN**, **KGCN**, and **XGBoost** models for audience score prediction  
        - Combine models via a **stacking ensemble** for improved accuracy  
        - Provide an interactive dashboard to:
            - Predict audience scores for specific films  
            - Explore dataset structure and success patterns  
            - Compare films and see similar titles  
        """
    )

    st.subheader("Architecture at a Glance")
    st.graphviz_chart(
        """
        digraph {
            rankdir=LR;
            node [shape=box, style=rounded];

            TMDB   -> Merge;
            RottenTomatoes -> Merge;
            YouTube -> Merge;

            Merge -> MongoDB;
            MongoDB -> "Feature Engineering";

            "Feature Engineering" -> GNN;
            "Feature Engineering" -> XGBoost;
            "Feature Engineering" -> KGCN;

            GNN -> Ensemble;
            XGBoost -> Ensemble;
            KGCN -> Ensemble;

            Ensemble -> "Audience Score Prediction";
        }
        """
    )


def home_page(db):
    st.title("Filmlytics Dashboard")
    st.subheader("Dataset Snapshot")
    st.markdown("---")

    stats = get_database_stats(db)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Movies", f"{stats['total']:,}")
    with col2:
        st.metric("Successful Movies", f"{stats['successful']:,}")
    with col3:
        rate = (
            stats["successful"] / stats["total"] * 100
            if stats["total"] > 0
            else 0.0
        )
        st.metric("Success Rate (TMDB)", f"{rate:.1f}%")

    st.markdown("---")
    st.subheader("Top Rated Movies (TMDB)")

    top_movies = get_top_movies(db, limit=10)

    for i, movie in enumerate(top_movies[:5], 1):
        with st.expander(
            f"{i}. {movie['title']} – {movie['tmdb_metrics']['vote_average']}/10"
        ):
            c1, c2 = st.columns([1, 2])
            with c1:
                poster = movie.get("content", {}).get("poster_url")
                if poster:
                    st.image(poster, width=140)
            with c2:
                st.write(
                    f"**Genres:** {', '.join(movie['production'].get('genres', []))}"
                )
                st.write(
                    f"**Runtime:** {movie['production'].get('runtime', 'N/A')} minutes"
                )
                st.write(
                    f"**Votes:** {movie['tmdb_metrics']['vote_count']:,}"
                )
                overview = movie.get("content", {}).get("overview")
                if overview:
                    st.write(f"**Overview:** {overview[:260]}...")


def data_pipeline_page():
    st.title("Data Pipeline")
    st.markdown("---")

    st.subheader("Data Sources")
    st.write(
        """
        **TMDB (≈66,233 films)**  
        - Genres, runtime, release dates, budget  
        - Production companies and countries  
        - Cast and crew metadata  
        - Vote averages and counts  

        **Rotten Tomatoes (≈6,800 films)**  
        - Audience scores (0–100)  
        - Critic scores (Tomatometer)  
        - Aggregated critic review sentiment  

        **YouTube API (≈42,156 trailers)**  
        - Trailer titles, upload dates, tags, categories  
        - Engagement: views, likes, comments, favorites  
        - Official vs. unofficial trailer filtering  
        """
    )

    st.subheader("Cleaning & Integration")
    st.write(
        """
        - Merged TMDB, Rotten Tomatoes, and YouTube trailers by IDs, titles, and release years  
        - Removed duplicate trailers using:
            - "Official/main trailer" filtering  
            - Earliest valid upload when multiple options exist  
        - Standardized numeric features and applied log-transform to heavy-tailed counts  
        - Extracted **sentiment scores** for descriptions / reviews using DistilBERT  
        - Computed gender representation metrics (e.g., % of female cast)  
        """
    )

    st.subheader("MongoDB Schema")
    st.graphviz_chart(
        """
        digraph {
            rankdir=TB;
            node [shape=box, style=rounded];

            Movie -> Production;
            Movie -> Metrics;
            Movie -> RottenTomatoes;
            Movie -> Trailer;

            Production -> Cast;
            Production -> Director;
        }
        """
    )


def modeling_overview_page():
    st.title("Modeling Overview")
    st.markdown("---")

    st.subheader("Graph Neural Network (GNN)")
    st.write(
        """
        - Nodes: ~66k films  
        - Edges: shared genres, directors, production countries, and gender-representation
          similarity (via k-nearest neighbors in a 10D diversity space)  
        - Architecture: two-layer **GraphSAGE** with batch normalization, dropout,
          and residual connections  
        - Training: semi-supervised on ~5k labeled films, with unlabeled nodes contributing
          via message passing  
        - Key takeaway: captures film-to-film similarity and global relational structure  
        """
    )

    st.graphviz_chart(
        """
        digraph {
            rankdir=LR;
            node [shape=box, style=rounded];

            "Movie A" -> "Movie B" [label="shared genre"];
            "Movie A" -> "Movie C" [label="same director"];
            "Movie A" -> "Movie D" [label="similar gender mix"];
            "Movie Graph" -> "GNN Encoder";
        }
        """
    )

    st.subheader("XGBoost")
    st.write(
        """
        - ~150 engineered features per film  
        - Feature groups:
            - Production: budget, runtime, US vs. non-US, release year  
            - Cast/director: historical performance, top actor/director indicators  
            - Trailer: engagement velocity, recency-weighted views/likes  
            - Diversity: gender balance and representation metrics  
        - Hyperparameters tuned via RandomizedSearchCV  
        - Strongest individual point predictor (lowest RMSE among single models)  
        """
    )

    st.subheader("Knowledge Graph Convolutional Network (KGCN)")
    st.write(
        """
        - Nodes: films, genres, directors, production companies, etc.  
        - Edges: semantic relations such as "directed_by", "has_genre", "produced_by"  
        - Learns relation-specific weights, allowing different edge types to have
          different predictive importance.  
        - Handles large, sparse graphs with edge-type normalization and sampling.  
        """
    )

    st.subheader("Model Performance (Summary)")
    perf_df = pd.DataFrame(
        {
            "Model": ["GNN", "KGCN", "XGBoost"],
            "RMSE": [0.197, 0.178, 0.177],
        }
    ).set_index("Model")
    st.bar_chart(perf_df)


def ensemble_model_page(artifacts):
    st.title("Ensemble Model")
    st.markdown("---")

    stacking_model = artifacts.get("stacking_model") if artifacts else None
    ensemble_meta = artifacts.get("ensemble_meta", {}) if artifacts else {}

    if stacking_model is not None:
        st.success("Stacking meta-learner loaded successfully.")
    else:
        st.warning(
            "Stacking model not found. The app will fall back to a weighted average of base models."
        )

    st.subheader("Ensemble Strategy")
    st.write(
        """
        The ensemble combines three complementary signals:

        - **GNN** – captures relational similarity across the film graph  
        - **KGCN** – captures typed knowledge-graph relationships  
        - **XGBoost** – captures nonlinear dependencies in engineered tabular features  

        When available, a **stacking meta-learner** (e.g., Gradient Boosting Regressor)
        takes `(GNN_pred, KGCN_pred, XGB_pred)` as input and outputs a final
        audience-score prediction.
        """
    )

    st.markdown("---")
    st.subheader("Stacking Performance")

    if stacking_model is not None:
        c1, c2, c3 = st.columns(3)
        with c1:
            model_name = ensemble_meta.get("meta_model", "Gradient Boosting")
            st.metric("Meta-Model", model_name)
        with c2:
            rmse = ensemble_meta.get("stacking_rmse", 0.1085)
            st.metric("Stacking RMSE", f"{rmse:.4f}")
        with c3:
            st.metric("Accuracy (±10%)", ensemble_meta.get("within_10", "80.2%"))

        st.info(
            """
            **How stacking works:**
            1. Each base model predicts an audience score.  
            2. These predictions form a 3D feature vector.  
            3. The meta-learner fits on ground-truth scores and learns how to optimally weight
               and combine base predictions.  
            4. At inference time, we pass base predictions through the meta-learner to obtain
               the final ensemble prediction.
            """
        )
    else:
        st.write(
            """
            The stacking model is not available in the current deployment environment.
            The app falls back to a **weighted average** of base model predictions,
            with default weights:
            - GNN: 0.33  
            - KGCN: 0.34  
            - XGBoost: 0.33  
            """
        )

    st.markdown("---")
    st.subheader("Individual Model Performance (Reference)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("GNN RMSE", "0.1929")
        st.caption("Graph-based similarity")
    with c2:
        st.metric("KGCN RMSE", "0.1709")
        st.caption("Relation-aware modeling")
    with c3:
        st.metric("XGBoost RMSE", "0.1097")
        st.caption("Feature-engineered gradient boosting")

    st.markdown("---")
    st.subheader("Coverage of Precomputed Predictions")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        n_gnn = (
            len(artifacts["gnn_preds"])
            if artifacts and artifacts.get("gnn_preds") is not None
            else 0
        )
        st.metric("GNN Predictions", f"{n_gnn:,}")
    with cc2:
        n_kgcn = (
            len(artifacts["kgcn_preds"])
            if artifacts and artifacts.get("kgcn_preds") is not None
            else 0
        )
        st.metric("KGCN Predictions", f"{n_kgcn:,}")
    with cc3:
        n_xgb = (
            len(artifacts["xgb_preds"])
            if artifacts and artifacts.get("xgb_preds") is not None
            else 0
        )
        st.metric("XGBoost Predictions", f"{n_xgb:,}")


def database_stats_page(db):
    st.title("Database Statistics")
    st.markdown("---")

    stats = get_database_stats(db)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Movies", f"{stats['total']:,}")
    with c2:
        st.metric("Successful", f"{stats['successful']:,}")
    with c3:
        st.metric("With RT Data", f"{stats['with_rotten_tomatoes']:,}")
    with c4:
        st.metric("With Trailers", f"{stats['with_trailers']:,}")

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Data Completeness")
        completeness_data = {
            "Category": ["Rotten Tomatoes", "Trailers", "Success Labels"],
            "Percentage": [
                (stats["with_rotten_tomatoes"] / stats["total"]) * 100
                if stats["total"] > 0
                else 0,
                (stats["with_trailers"] / stats["total"]) * 100
                if stats["total"] > 0
                else 0,
                (stats["successful"] / stats["total"]) * 100
                if stats["total"] > 0
                else 0,
            ],
        }
        df_complete = pd.DataFrame(completeness_data)
        fig = px.bar(
            df_complete,
            x="Category",
            y="Percentage",
            title="Data Completeness (%)",
            color="Percentage",
            color_continuous_scale="greens",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Top Genres")
        pipeline_genres = [
            {"$unwind": "$production.genres"},
            {"$group": {"_id": "$production.genres", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 5},
        ]
        for doc in db.movies.aggregate(pipeline_genres):
            st.write(f"- {doc['_id']}: {doc['count']:,}")


def movie_search_page(db, artifacts):
    st.title("Movie Search & Prediction")
    st.markdown("---")

    all_titles = get_all_movie_titles(db)
    selected_title = st.selectbox(
        "Select or type a movie title:",
        options=["-- Select a Movie --"] + all_titles,
        index=0,
    )

    search_query = (
        selected_title if selected_title != "-- Select a Movie --" else None
    )

    if search_query:
        movie = search_movie(db, search_query)
        if movie:
            st.success(f"Found: {movie['title']}")

            ensemble_score, breakdown = predict_ensemble(movie, artifacts)

            st.markdown("---")
            st.subheader("Ensemble Audience Score Prediction")
            if not np.isnan(ensemble_score):
                st.metric(
                    "Predicted Audience Score",
                    f"{ensemble_score*100:.1f}%",
                )
                st.caption("Predicted Rotten Tomatoes audience score")

                st.markdown("#### Model Breakdown")
                b1, b2, b3 = st.columns(3)
                with b1:
                    score = breakdown["gnn"]
                    st.metric(
                        "GNN Prediction",
                        f"{score*100:.1f}%"
                        if not np.isnan(score)
                        else "N/A",
                    )
                with b2:
                    score = breakdown["kgcn"]
                    st.metric(
                        "KGCN Prediction",
                        f"{score*100:.1f}%"
                        if not np.isnan(score)
                        else "N/A",
                    )
                with b3:
                    score = breakdown["xg"]
                    st.metric(
                        "XGBoost Prediction",
                        f"{score*100:.1f}%"
                        if not np.isnan(score)
                        else "N/A",
                    )
            else:
                st.warning(
                    "Unable to compute ensemble prediction for this title (missing base model predictions)."
                )

            st.markdown("---")

            c1, c2 = st.columns([1, 2])
            with c1:
                poster = movie.get("content", {}).get("poster_url")
                if poster:
                    st.image(poster, width=240)
            with c2:
                st.subheader(movie["title"])
                st.write(f"**TMDB ID:** {movie['tmdb_id']}")
                st.write(
                    f"**Genres:** {', '.join(movie['production'].get('genres', []))}"
                )
                st.write(
                    f"**Runtime:** {movie['production'].get('runtime', 'N/A')} minutes"
                )
                st.write(
                    f"**Budget:** ${movie['production'].get('budget', 0):,.0f}"
                )
                st.write(
                    f"**Release Date:** {movie['release_info'].get('tmdb_release_date', 'N/A')}"
                )

                st.markdown("### Ratings")
                rc1, rc2, rc3 = st.columns(3)
                with rc1:
                    st.metric(
                        "TMDB Rating",
                        f"{movie['tmdb_metrics'].get('vote_average', 'N/A')}/10",
                    )
                with rc2:
                    rt_crit = movie.get("rotten_tomatoes", {}).get(
                        "critic_score"
                    )
                    st.metric("RT Critics", rt_crit if rt_crit else "N/A")
                with rc3:
                    rt_aud = movie.get("rotten_tomatoes", {}).get(
                        "audience_score"
                    )
                    st.metric("RT Audience", rt_aud if rt_aud else "N/A")

                if movie["tmdb_metrics"].get("is_successful"):
                    st.success("Classified as: SUCCESSFUL (vote_average > 6)")
                else:
                    st.error("Classified as: NOT SUCCESSFUL (vote_average ≤ 6)")

            st.markdown("### Similar Movies (Genre-Based)")
            similar = get_similar_movies(db, movie["tmdb_id"], limit=5)
            if similar:
                cols = st.columns(len(similar))
                for idx, sim_movie in enumerate(similar):
                    with cols[idx]:
                        st.write(f"**{sim_movie['title']}**")
                        st.write(
                            f"{sim_movie['tmdb_metrics']['vote_average']}/10"
                        )
            else:
                st.write("No similar movies found.")
        else:
            st.error(f"No movie found matching '{search_query}'")

    st.markdown("---")
    st.subheader("Browse by Genre")

    genres = get_all_genres(db)
    selected_genre = st.selectbox("Select a genre:", genres)

    if selected_genre:
        genre_movies = get_movies_by_genre(db, selected_genre, limit=10)
        st.write(f"Showing top {len(genre_movies)} {selected_genre} movies:")
        for movie in genre_movies:
            st.write(
                f"**{movie['title']}** – {movie['tmdb_metrics']['vote_average']}/10"
            )


def compare_movies_page(db, artifacts):
    st.title("Compare Movies")
    st.markdown("---")

    titles = get_all_movie_titles(db)
    c1, c2 = st.columns(2)
    movie1_title = c1.selectbox("Select first movie:", titles, index=0)
    movie2_title = c2.selectbox("Select second movie:", titles, index=1)

    if movie1_title and movie2_title:
        m1 = search_movie(db, movie1_title)
        m2 = search_movie(db, movie2_title)

        if m1 is None or m2 is None:
            st.error(
                "One or both selected movies could not be found in the database."
            )
            return

        st.subheader("Basic Feature Comparison")

        comparison = pd.DataFrame(
            {
                "Feature": ["TMDB Score", "Vote Count", "Runtime", "Budget"],
                movie1_title: [
                    m1["tmdb_metrics"]["vote_average"],
                    m1["tmdb_metrics"]["vote_count"],
                    m1["production"].get("runtime", None),
                    m1["production"].get("budget", None),
                ],
                movie2_title: [
                    m2["tmdb_metrics"]["vote_average"],
                    m2["tmdb_metrics"]["vote_count"],
                    m2["production"].get("runtime", None),
                    m2["production"].get("budget", None),
                ],
            }
        )
        st.table(comparison)

        st.subheader("Predicted Audience Score Comparison")
        pred1, _ = predict_ensemble(m1, artifacts)
        pred2, _ = predict_ensemble(m2, artifacts)

        pc1, pc2 = st.columns(2)
        pc1.metric(
            movie1_title,
            f"{pred1*100:.1f}%"
            if not np.isnan(pred1)
            else "N/A",
        )
        pc2.metric(
            movie2_title,
            f"{pred2*100:.1f}%"
            if not np.isnan(pred2)
            else "N/A",
        )


def analytics_page(db):
    st.title("Visualizations & Analytics")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(
        ["Genre Distribution", "Rating Distribution", "Success Trends"]
    )

    with tab1:
        st.subheader("Genre Distribution")
        fig = create_genre_distribution_chart(db)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No sufficient genre data to render this chart.")

    with tab2:
        st.subheader("Rating Distribution")
        fig = create_rating_distribution(db)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No sufficient rating data to render this chart.")

    with tab3:
        st.subheader("Success Rate Over Time")
        fig = create_success_over_time(db)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data for time-series analysis.")


def team_page():
    st.title("Team & Acknowledgments")
    st.markdown("---")

    st.subheader("Team 15 – Cinemaniacs")
    st.write(
        """
        - Angela Cottone  
        - Nidhi Deshmukh  
        - Dylan Sidhu  
        - Matthew Ward  
        - Clara Wei  
        """
    )

    st.subheader("Technologies")
    st.write(
        """
        - Streamlit  
        - MongoDB  
        - PyTorch Geometric  
        - XGBoost  
        - Hugging Face Transformers  
        - Python / Pandas / NumPy  
        """
    )

    st.subheader("Data Sources")
    st.write(
        """
        - TMDB API  
        - Rotten Tomatoes Website  
        - YouTube Data API  
        """
    )

    st.subheader("Acknowledgments")
    st.write(
        """
        We thank the STA 160 instructional team for guidance, and note that AI tools
        (including ChatGPT, Google Gemini, and GitHub Copilot) were used to assist
        with ideation, code prototyping, and documentation.
        """
    )


# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------

def main():
    db = get_database_connection()
    if db is None:
        st.error(
            "Unable to connect to database. Please verify your MongoDB credentials."
        )
        st.stop()

    artifacts, error = load_ensemble_artifacts()
    if error:
        st.error(f"Ensemble artifacts failed to load: {error}")
        st.stop()
    else:
        st.sidebar.success("Model artifacts loaded")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Introduction",
            "Home",
            "Modeling",
            "Movie Search",
            "Compare Movies",
            "Analytics",
            "Team & Acknowledgments",
        ],
    )

    modeling_subpage = None
    if page == "Modeling":
        with st.sidebar.expander("Modeling & Data Views", expanded=True):
            modeling_subpage = st.radio(
                "Select view",
                [
                    "Modeling Overview",
                    "Data Pipeline",
                    "Ensemble Model",
                    "Database Stats",
                ],
                label_visibility="collapsed",
            )

    # Route pages
    if page == "Introduction":
        introduction_page()
    elif page == "Home":
        home_page(db)
    elif page == "Modeling":
        if modeling_subpage == "Modeling Overview":
            modeling_overview_page()
        elif modeling_subpage == "Data Pipeline":
            data_pipeline_page()
        elif modeling_subpage == "Ensemble Model":
            ensemble_model_page(artifacts)
        elif modeling_subpage == "Database Stats":
            database_stats_page(db)
    elif page == "Movie Search":
        movie_search_page(db, artifacts)
    elif page == "Compare Movies":
        compare_movies_page(db, artifacts)
    elif page == "Analytics":
        analytics_page(db)
    elif page == "Team & Acknowledgments":
        team_page()

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; font-size: 0.9rem; color: #666;'>
            <p>Filmlytics – STA 160 Project – Team 15</p>
            <p>Graph-Based Ensemble Prediction of Movie Audience Scores</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()