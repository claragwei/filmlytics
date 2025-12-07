import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import certifi
import os
import joblib  # For loading XGBoost model and scaler
import json    # For loading feature list / ensemble meta
import pickle  # For loading stacking model

# =============================================================================
# CONFIG & GLOBAL STYLE
# =============================================================================

# MongoDB Connection – uses secrets on cloud, falls back to hard-coded URI locally
MONGODB_URI = st.secrets.get(
    "MONGODB_URI",
    "mongodb+srv://cinemaniacs:filmlytics@filmlytics.1emhcue.mongodb.net/?appName=filmlytics"
)

# Page Configuration
st.set_page_config(
    page_title="Predicting Movie Audience Scores Using Graph-Based Modeling",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple custom CSS for a more modern feel
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #666666;
        margin-bottom: 1.2rem;
    }
    .hero-buttons-row button[kind="secondary"] {
        border-radius: 999px !important;
        border: 1px solid #cccccc !important;
    }
    .card {
        padding: 1rem 1.2rem;
        border-radius: 0.75rem;
        background-color: #11182711;
        border: 1px solid #e5e7eb20;
        margin-bottom: 1rem;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 0.5rem;
        margin-bottom: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# DATABASE CONNECTION
# =============================================================================

@st.cache_resource
def get_database_connection():
    """Connect to MongoDB and cache the connection."""
    try:
        client = MongoClient(
            MONGODB_URI,
            server_api=ServerApi('1'),
            tlsCAFile=certifi.where()
        )
        db = client['cinemaniacs']
        # Test connection
        db.movies.count_documents({})
        return db
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

# =============================================================================
# ARTIFACT LOADING AND ENSEMBLE PREDICTION SETUP
# =============================================================================

@st.cache_resource
def load_ensemble_artifacts():
    """
    Load all models, scalers, and prediction dataframes including stacking model.
    Uses precomputed predictions from:
      - gnn_preds_all_movies.csv
      - kgcn_preds_all_movies.csv
      - xgb_preds_all_movies.csv
    And an optional stacking_meta_model.pkl (+ ensemble_weights.json).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    artifact_dir = os.path.join(script_dir, 'model_artifacts')

    artifacts = {}
    required_files = {
        # XGBoost base model + feature list (not used live, but kept for completeness)
        'xg_model': 'xgboost_base_model.pkl',
        'xg_features': 'xg_feature_columns.json',
        # Shared scaler (if needed later)
        'scaler': 'movie_feature_scaler_diversity.pkl',
        # Precomputed predictions
        'gnn_preds': 'gnn_preds_all_movies.csv',
        'kgcn_preds': 'kgcn_preds_all_movies.csv',
        'xgb_preds': 'xgb_preds_all_movies.csv',
    }

    for key, filename in required_files.items():
        path = os.path.join(artifact_dir, filename)
        if not os.path.exists(path):
            st.warning(f"Missing required file: {filename}. Some predictions will be unavailable.")
            artifacts[key] = None
            continue

        try:
            if filename.endswith('.pkl'):
                artifacts[key] = joblib.load(path)
            elif filename.endswith('.json'):
                with open(path, 'r') as f:
                    artifacts[key] = json.load(f)
            elif filename.endswith('.csv'):
                df = pd.read_csv(path)
                df['tmdb_id'] = pd.to_numeric(df['tmdb_id'], errors='coerce').fillna(0).astype(int)

                if 'pred_audience_score' in df.columns:
                    artifacts[key] = df.set_index('tmdb_id')['pred_audience_score']
                elif 'predicted_audience_score' in df.columns:
                    artifacts[key] = df.set_index('tmdb_id')['predicted_audience_score']
                elif 'ensemble_pred' in df.columns:
                    artifacts[key] = df.set_index('tmdb_id')['ensemble_pred']
                else:
                    artifacts[key] = df.set_index('tmdb_id').iloc[:, 0]
        except Exception as e:
            return None, f"Error loading {filename}: {e}"

    # Stacking meta-model (rebuilt in the same sklearn version as runtime)
    stacking_path = os.path.join(artifact_dir, 'stacking_meta_model.pkl')
    if os.path.exists(stacking_path):
        try:
            with open(stacking_path, 'rb') as f:
                artifacts['stacking_model'] = pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load stacking model: {e}")
            artifacts['stacking_model'] = None
    else:
        artifacts['stacking_model'] = None

    # Ensemble metadata
    meta_path = os.path.join(artifact_dir, 'ensemble_weights.json')
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                artifacts['ensemble_meta'] = json.load(f)
        except Exception:
            artifacts['ensemble_meta'] = {}
    else:
        artifacts['ensemble_meta'] = {}

    # Fallback weights for weighted average if stacking unavailable
    artifacts['ensemble_weights'] = {'gnn': 0.33, 'kgcn': 0.34, 'xg': 0.33}

    return artifacts, None

def safe_get_prediction(preds, tmdb_id):
    """Safely get a prediction value from a pandas Series (or None)."""
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
    Uses stacking meta-learner to combine GNN, KGCN, and XGBoost predictions.
    Falls back to weighted average if stacking model is not available.

    Returns: ensemble_prediction (0-1), prediction_breakdown (dict)
    """
    tmdb_id = movie_data.get('tmdb_id')
    if tmdb_id is None:
        return np.nan, {}

    gnn_pred = safe_get_prediction(artifacts.get('gnn_preds'), tmdb_id)
    kgcn_pred = safe_get_prediction(artifacts.get('kgcn_preds'), tmdb_id)
    xgb_pred = safe_get_prediction(artifacts.get('xgb_preds'), tmdb_id)

    predictions = {
        'gnn': gnn_pred,
        'kgcn': kgcn_pred,
        'xg': xgb_pred
    }

    stacking_model = artifacts.get('stacking_model')
    has_all_preds = not (np.isnan(gnn_pred) or np.isnan(kgcn_pred) or np.isnan(xgb_pred))

    # Preferred: meta-learner
    if stacking_model is not None and has_all_preds:
        X = np.array([[gnn_pred, kgcn_pred, xgb_pred]])
        ensemble_pred = float(np.clip(stacking_model.predict(X)[0], 0, 1))
        return ensemble_pred, predictions

    # Fallback: weighted average
    weights = artifacts.get('ensemble_weights', {'gnn': 0.33, 'kgcn': 0.34, 'xg': 0.33})
    valid_preds = {k: v for k, v in predictions.items() if not np.isnan(v)}

    if not valid_preds:
        tmdb_avg = movie_data.get('tmdb_metrics', {}).get('vote_average')
        if tmdb_avg is not None:
            return float(tmdb_avg) / 10.0, predictions
        return np.nan, predictions

    valid_keys = list(valid_preds.keys())
    total_valid_weight = sum(weights.get(k, 0) for k in valid_keys)

    if total_valid_weight == 0:
        ensemble_pred = np.mean(list(valid_preds.values()))
    else:
        ensemble_pred = sum(
            valid_preds[k] * (weights[k] / total_valid_weight)
            for k in valid_keys
        )

    ensemble_pred = np.clip(ensemble_pred, 0.0, 1.0)
    return ensemble_pred, predictions

# =============================================================================
# DATA QUERY FUNCTIONS
# =============================================================================

@st.cache_data
def get_all_movie_titles(_db):
    """Get a list of UNIQUE movie titles, sorted by vote count (popularity)."""
    try:
        pipeline = [
            {"$match": {
                "title": {"$ne": None},
                "tmdb_id": {"$ne": None},
                "tmdb_metrics.vote_count": {"$gte": 1}
            }},
            {"$group": {
                "_id": "$tmdb_id",
                "title": {"$first": "$title"},
                "vote_count": {"$max": "$tmdb_metrics.vote_count"}
            }},
            {"$sort": {"vote_count": -1}},
            {"$project": {"_id": 0, "title": 1}}
        ]
        unique_titles = list(_db.movies.aggregate(pipeline))
        return [doc['title'] for doc in unique_titles]
    except Exception as e:
        st.error(f"Error fetching unique movie titles: {e}")
        return []

def search_movie(db, title):
    movie = db.movies.find_one({"title": {"$regex": f"^{title}$", "$options": "i"}})
    if not movie:
        movie = db.movies.find_one({"title": {"$regex": title, "$options": "i"}})
    return movie

def get_top_movies(db, limit=50, min_votes=1000):
    query = {
        "tmdb_metrics.vote_count": {"$gte": min_votes},
        "tmdb_metrics.vote_average": {"$ne": None}
    }
    return list(db.movies.find(query).sort("tmdb_metrics.vote_average", -1).limit(limit))

def get_similar_movies(db, tmdb_id, limit=10, min_votes=1000):
    """Simple similarity via exact genre set match + vote filter."""
    movie = db.movies.find_one({"tmdb_id": tmdb_id})
    if not movie:
        return []
    genres = movie['production'].get('genres', [])
    if not genres:
        return []
    genre_count = len(genres)

    pipeline = [
        {"$match": {
            "production.genres": {
                "$all": genres,
                "$size": genre_count
            },
            "tmdb_id": {"$ne": tmdb_id},
            "tmdb_metrics.vote_count": {"$gte": min_votes},
            "tmdb_metrics.vote_average": {"$ne": None}
        }},
        {"$group": {
            "_id": "$tmdb_id",
            "unique_movie": {"$first": "$$ROOT"}
        }},
        {"$replaceRoot": {"newRoot": "$unique_movie"}},
        {"$sort": {"tmdb_metrics.vote_average": -1}},
        {"$limit": limit}
    ]
    return list(db.movies.aggregate(pipeline))

def get_database_stats(db):
    total = db.movies.count_documents({})
    successful = db.movies.count_documents({"tmdb_metrics.is_successful": True})
    with_rt = db.movies.count_documents({"rotten_tomatoes.has_rt_url": True})
    with_trailer = db.movies.count_documents({"trailer.trailer_url_youtube": {"$ne": None}})
    
    return {
        "total": total,
        "successful": successful,
        "with_rotten_tomatoes": with_rt,
        "with_trailers": with_trailer
    }

def get_all_genres(db):
    genres = db.movies.distinct("production.genres")
    return sorted([g for g in genres if g])

def get_movies_by_genre(db, genre, limit=20, min_votes=1000):
    pipeline = [
        {"$match": {
            "production.genres": genre,
            "tmdb_metrics.vote_count": {"$gte": min_votes},
            "tmdb_metrics.vote_average": {"$ne": None}
        }},
        {"$group": {
            "_id": "$tmdb_id",
            "unique_movie": {"$first": "$$ROOT"}
        }},
        {"$replaceRoot": {"newRoot": "$unique_movie"}},
        {"$sort": {"tmdb_metrics.vote_average": -1}},
        {"$limit": limit}
    ]
    return list(db.movies.aggregate(pipeline))

# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def create_genre_distribution_chart(db):
    pipeline = [
        {"$unwind": "$production.genres"},
        {"$group": {"_id": "$production.genres", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 15}
    ]
    results = list(db.movies.aggregate(pipeline))
    df = pd.DataFrame(results)
    if df.empty:
        return None
    df.columns = ['Genre', 'Count']
    fig = px.bar(
        df,
        x='Genre',
        y='Count',
        title='Top 15 Movie Genres',
        color='Count',
        color_continuous_scale='viridis'
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def create_rating_distribution(db):
    movies = list(db.movies.find(
        {"tmdb_metrics.vote_average": {"$ne": None}},
        {"tmdb_metrics.vote_average": 1}
    ).limit(5000))
    ratings = [m['tmdb_metrics']['vote_average'] for m in movies]
    if not ratings:
        return None
    fig = px.histogram(
        ratings,
        nbins=50,
        title='Distribution of TMDB Ratings',
        labels={'value': 'Rating', 'count': 'Number of Movies'},
        color_discrete_sequence=['#2563eb']
    )
    fig.update_layout(showlegend=False)
    return fig

def create_success_over_time(db):
    pipeline = [
        {"$match": {"release_info.tmdb_release_date": {"$ne": None, "$regex": "^[0-9]{4}"}}},
        {"$project": {
            "year": {"$substr": ["$release_info.tmdb_release_date", 0, 4]},
            "is_successful": "$tmdb_metrics.is_successful"
        }},
        {"$group": {
            "_id": "$year",
            "total": {"$sum": 1},
            "successful": {"$sum": {"$cond": ["$is_successful", 1, 0]}}
        }},
        {"$project": {
            "year": "$_id",
            "success_rate": {"$multiply": [{"$divide": ["$successful", "$total"]}, 100]}
        }},
        {"$sort": {"year": 1}}
    ]
    results = list(db.movies.aggregate(pipeline))
    df = pd.DataFrame(results)
    if not df.empty and len(df) > 10:
        df = df[df['year'].astype(int) >= 2010]
        if df.empty:
            return None
        fig = px.line(
            df,
            x='year',
            y='success_rate',
            title='Movie Success Rate Over Time',
            labels={'year': 'Year', 'success_rate': 'Success Rate (%)'},
            markers=True
        )
        fig.update_layout(hovermode='x unified')
        return fig
    return None

def create_similarity_graph_figure(db, center_movie, max_neighbors=15):
    """
    Build a simple similarity graph around `center_movie`:
    - Center node = selected movie
    - Neighbors = top movies in one of its genres
    Rendered as a Plotly scatter+edges.
    """
    if center_movie is None:
        return None

    genres = center_movie.get("production", {}).get("genres", [])
    if not genres:
        return None

    # Use first genre for a small local neighborhood
    genre = genres[0]
    neighbors = get_movies_by_genre(db, genre, limit=max_neighbors + 1)
    if not neighbors:
        return None

    G = nx.Graph()
    center_id = center_movie["tmdb_id"]
    center_title = center_movie["title"]

    G.add_node(center_id, title=center_title, is_center=True)

    for m in neighbors:
        mid = m["tmdb_id"]
        if mid == center_id:
            continue
        G.add_node(mid, title=m["title"], is_center=False)
        G.add_edge(center_id, mid)

    # Layout
    pos = nx.spring_layout(G, seed=42, k=0.6)

    # Build edge traces
    edge_x = []
    edge_y = []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color="#9ca3af"),
        hoverinfo="none",
        mode="lines"
    )

    # Node trace
    node_x = []
    node_y = []
    node_text = []
    node_color = []

    for node_id, data in G.nodes(data=True):
        x, y = pos[node_id]
        node_x.append(x)
        node_y.append(y)
        node_text.append(data.get("title", str(node_id)))
        node_color.append("#fbbf24" if data.get("is_center") else "#60a5fa")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hoverinfo="text",
        marker=dict(
            size=[16 if c == "#fbbf24" else 10 for c in node_color],
            color=node_color,
            line=dict(width=1, color="#111827")
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=f"Similarity Graph Around: {center_title}",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, t=40, b=10),
        height=500,
    )
    return fig

# =============================================================================
# PAGE FUNCTIONS
# =============================================================================

def home_page(db):
    # Title + subtitle
    st.markdown("<div class='main-title'>Predicting Movie Audience Scores Using Graph-Based Modeling</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Filmytics — Combining graph learning, metadata, and ensemble modeling for film audience prediction.</div>", unsafe_allow_html=True)

    # Hero navigation buttons
    st.markdown("<div class='hero-buttons-row'>", unsafe_allow_html=True)
    btn_cols = st.columns(4)
    buttons = [
        ("Home", "Home"),
        ("Movie Search", "Movie Search"),
        ("Compare Movies", "Compare Movies"),
        ("Analytics Dashboard", "Analytics Dashboard"),
        ("Modeling", "Modeling"),
        ("Visual Graph Explorer", "Visual Graph Explorer"),
        ("Acknowledgements", "Acknowledgements"),
    ]

    for (label, target), col in zip(buttons[:4], btn_cols):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state["page"] = target
                st.rerun()

    btn_cols2 = st.columns(3)
    for (label, target), col in zip(buttons[4:], btn_cols2):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state["page"] = target
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

    # ===============================
    # WHAT THIS PROJECT IS & WHY IT MATTERS
    # ===============================
    st.markdown("## Filmytics — Predicting Audience Scores")

    st.write(
        """
**Filmytics** is a data-driven project that predicts audience scores for movies *before* release.  
Studios, distributors, and creators depend on audience reception to shape marketing, streaming deals, and franchise decisions.

For movie fans and researchers, understanding what drives audience opinion reveals which films resonate most and whether representation (e.g., gender balance in cast) affects outcomes.

Overall, Filmytics gathers public film data, learns patterns from past movies, and estimates how audiences will react to new releases.
        """
    )

    # ===============================
    # OUR APPROACH
    # ===============================
    st.markdown("## Our Approach")

    st.write(
        """
1. **Data collection from three major sources**  
   - **TMDB**: metadata (runtime, cast, genres, budget, release date)  
   - **Rotten Tomatoes**: critic & audience scores, review texts  
   - **YouTube API**: trailer views, likes, engagement metrics  

   → Final dataset covers **≈66,000 films (2010–2025)**.

2. **Data cleaning & merging** into structured MongoDB documents  
   - Remove duplicates  
   - Normalize numeric fields (log budgets, scaled metrics)  
   - Compute new metrics: trailer sentiment, female cast share  

3. **Feature Engineering**  
   - ~150 features: metadata, engagement signals, critic cues, representation features  
   - Normalized + encoded for model performance  

4. **Train multiple complementary models**  
   - **GNN** → similarity relationships between films  
   - **KGCN** → entity-level graph of films, people, genres  
   - **XGBoost** → rich tabular baseline  

5. **Stacking Ensemble**  
   - Gradient Boosting meta-learner  
   - Learns when to trust each model  
"""
    )

    st.markdown("### Workflow Diagram (Placeholder)")
    st.info("Insert workflow diagram here:\nTMDB + RT + YouTube → Clean/Merge → Feature Engineering → {GNN, KGCN, XGBoost} → Stacking Meta-Learner → Audience Score Prediction")

    # ===============================
    # KEY MODELING STEPS
    # ===============================
    st.markdown("## Key Modeling Steps")

    with st.expander("Step 1 — Data & Preprocessing"):
        st.write(
            """
- Join using TMDB IDs, resolve duplicates  
- Clean trailers & compute sentiment  
- Extract representation features (female cast %, billing alignment)  
- Store cleaned dataset in MongoDB  
            """
        )

    with st.expander("Step 2 — Feature Engineering"):
        st.write(
            """
Feature groups include:  
- Film metadata (runtime, budget, genres)  
- Social cues (views, likes, recency)  
- Critic-derived features  
- Diversity & representation indicators  
            """
        )

    with st.expander("Step 3 — Base Models"):
        st.write(
            """
**GNN:** Movie similarity graph using genres, directors, countries, representation alignment  
**KGCN:** Entity graph including cast, crew, companies  
**XGBoost:** 150+ engineered tabular features  
            """
        )

    with st.expander("Step 4 — Stacking Ensemble"):
        st.write(
            """
We combine predictions using a **Gradient Boosting meta-learner**, improving accuracy significantly  
beyond simple averaging.  
            """
        )

    # ===============================
    # MAIN FINDINGS & RESULTS
    # ===============================
    st.markdown("## Main Findings and Results")

    st.write(
        """
- XGBoost performed best among single models → **RMSE ≈ 0.110**  
- KGCN second-best → **RMSE ≈ 0.171**  
- GNN third → **RMSE ≈ 0.195**  
- Stacking ensemble improved performance to **RMSE ≈ 0.1085**, a large improvement over a simple weighted average (0.1452).  
- **80.2%** of predictions fall within ±10 percentage points of the true audience score.
        """
    )

    st.markdown("### Results Table (Placeholder)")
    st.info("Insert table of key metrics here — RMSE, MAE, coverage, % within ±10%, etc.")

    # ===============================
    # WHAT THIS MEANS & NEXT STEPS
    # ===============================
    st.markdown("## What This Means & Next Steps")

    st.write(
        """
This system can help studios:  
- Estimate audience reception early  
- Identify where representation correlates with performance  
- Prioritize marketing spend for films likely to underperform  

**Future Work**  
- Include social media trends (Twitter / TikTok)  
- Add fine-grained cast/crew diversity measures  
- Expand to multilingual markets  
        """
    )

    st.markdown("---")

    # ===============================
    # DATABASE STATISTICS (KEPT SAME)
    # ===============================
    stats = get_database_stats(db)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Movies", f"{stats['total']:,}")
    with c2:
        st.metric("With Audience Scores", f"{stats['with_rotten_tomatoes']:,}")
    with c3:
        success_rate = (stats['successful'] / stats['total'] * 100) if stats['total'] > 0 else 0
        st.metric("Success-Labeled %", f"{success_rate:.1f}%")

    st.subheader("Top Rated Movies")
    top_movies = get_top_movies(db, limit=10)
    for i, movie in enumerate(top_movies[:5], 1):
        with st.expander(f"{i}. {movie['title']} – TMDB {movie['tmdb_metrics']['vote_average']}/10"):
            c1, c2 = st.columns([1, 2])
            with c1:
                poster = movie.get('content', {}).get('poster_url')
                if poster:
                    st.image(poster, width=150)
            with c2:
                st.write(f"**Genres:** {', '.join(movie['production'].get('genres', []))}")
                st.write(f"**Runtime:** {movie['production'].get('runtime', 'N/A')} minutes")
                st.write(f"**Votes:** {movie['tmdb_metrics']['vote_count']:,}")
                overview = movie.get('content', {}).get('overview')
                if overview:
                    st.write(f"{overview[:260]}...")  # trim for display

def movie_search_page(db, artifacts):
    st.title("Movie Search & Audience Score Prediction")

    all_titles = get_all_movie_titles(db)
    selected_title = st.selectbox(
        "Select or type a movie title:",
        options=["-- Select a Movie --"] + all_titles,
        index=0
    )
    search_query = selected_title if selected_title != "-- Select a Movie --" else None

    movie = None
    if search_query:
        movie = search_movie(db, search_query)
        if movie:
            st.success(f"Found: {movie['title']}")
            ensemble_score, breakdown = predict_ensemble(movie, artifacts)

            st.markdown("---")
            st.subheader("Ensemble Audience Score Prediction")

            if not np.isnan(ensemble_score):
                st.success(f"Predicted Audience Score: {ensemble_score * 100:.1f}%")
                b_col1, b_col2, b_col3 = st.columns(3)
                with b_col1:
                    score = breakdown['gnn']
                    st.metric("GNN", f"{score * 100:.1f}%" if not np.isnan(score) else "N/A")
                with b_col2:
                    score = breakdown['kgcn']
                    st.metric("KGCN", f"{score * 100:.1f}%" if not np.isnan(score) else "N/A")
                with b_col3:
                    score = breakdown['xg']
                    st.metric("XGBoost", f"{score * 100:.1f}%" if not np.isnan(score) else "N/A")
            else:
                st.warning("Cannot generate ensemble prediction (missing base-model predictions for this movie).")

            st.markdown("---")

            col1, col2 = st.columns([1, 2])
            with col1:
                poster = movie.get('content', {}).get('poster_url')
                if poster:
                    st.image(poster, width=250)
            with col2:
                st.subheader(movie['title'])
                st.write(f"**TMDB ID:** {movie['tmdb_id']}")
                st.write(f"**Genres:** {', '.join(movie['production'].get('genres', []))}")
                st.write(f"**Runtime:** {movie['production'].get('runtime', 'N/A')} minutes")
                st.write(f"**Budget:** ${movie['production'].get('budget', 0):,.0f}")
                st.write(f"**Release Date:** {movie['release_info'].get('tmdb_release_date', 'N/A')}")

                st.markdown("**Ratings**")
                r1, r2, r3 = st.columns(3)
                with r1:
                    st.metric("TMDB", f"{movie['tmdb_metrics'].get('vote_average', 'N/A')}/10")
                with r2:
                    rt_crit = movie.get('rotten_tomatoes', {}).get('critic_score')
                    st.metric("RT Critics", rt_crit if rt_crit else "N/A")
                with r3:
                    rt_aud = movie.get('rotten_tomatoes', {}).get('audience_score')
                    st.metric("RT Audience", rt_aud if rt_aud else "N/A")

                if movie['tmdb_metrics'].get('is_successful'):
                    st.success("Classified as: SUCCESSFUL")
                else:
                    st.error("Classified as: NOT SUCCESSFUL")

            st.markdown("### Similar Movies")
            similar = get_similar_movies(db, movie['tmdb_id'], limit=5)
            if similar:
                cols = st.columns(len(similar))
                for idx, sim_movie in enumerate(similar):
                    with cols[idx]:
                        st.write(f"**{sim_movie['title']}**")
                        st.write(f"TMDB: {sim_movie['tmdb_metrics']['vote_average']}/10")
            else:
                st.write("No similar movies found.")

            # Inline Visual Graph Explorer for this movie
            st.markdown("---")
            st.subheader("Visual Graph Explorer (Local Neighborhood)")
            fig = create_similarity_graph_figure(db, movie, max_neighbors=15)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough neighborhood information to build a similarity graph for this film.")

        else:
            st.error(f"No movie found matching '{search_query}'")

    st.markdown("---")
    st.subheader("Browse by Genre")
    genres = get_all_genres(db)
    if genres:
        selected_genre = st.selectbox("Select a genre:", genres)
        if selected_genre:
            genre_movies = get_movies_by_genre(db, selected_genre, limit=10)
            st.write(f"Showing top {len(genre_movies)} {selected_genre} movies:")
            for m in genre_movies:
                st.write(f"**{m['title']}** – TMDB {m['tmdb_metrics']['vote_average']}/10")

def compare_movies_page(db, artifacts):
    st.title("Compare Movies")

    titles = get_all_movie_titles(db)
    col1, col2 = st.columns(2)
    movie1_title = col1.selectbox("Select first movie:", ["-- Select --"] + titles)
    movie2_title = col2.selectbox("Select second movie:", ["-- Select --"] + titles)

    if movie1_title != "-- Select --" and movie2_title != "-- Select --":
        m1 = search_movie(db, movie1_title)
        m2 = search_movie(db, movie2_title)

        if m1 is None or m2 is None:
            st.error("One or both selected movies could not be found.")
            return

        st.subheader("Basic Feature Comparison")
        comparison = pd.DataFrame({
            "Feature": ["TMDB Score", "Vote Count", "Runtime (min)", "Budget ($)"],
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
            ]
        })
        st.table(comparison)

        st.subheader("Ensemble Audience Score Comparison")
        pred1, _ = predict_ensemble(m1, artifacts)
        pred2, _ = predict_ensemble(m2, artifacts)
        c1, c2 = st.columns(2)
        with c1:
            st.metric(movie1_title, f"{pred1 * 100:.1f}%" if not np.isnan(pred1) else "N/A")
        with c2:
            st.metric(movie2_title, f"{pred2 * 100:.1f}%" if not np.isnan(pred2) else "N/A")

def analytics_page(db):
    st.title("Analytics Dashboard")

    tab1, tab2, tab3 = st.tabs(["Genre Analysis", "Rating Distribution", "Success Trends"])
    with tab1:
        st.subheader("Genre Distribution")
        fig = create_genre_distribution_chart(db)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data to compute genre distribution.")
    with tab2:
        st.subheader("TMDB Rating Distribution")
        fig = create_rating_distribution(db)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data to compute rating distribution.")
    with tab3:
        st.subheader("Success Rate Over Time")
        fig = create_success_over_time(db)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data for time series analysis.")

def modeling_page(artifacts):
    st.title("Modeling & Ensemble Framework")

    st.markdown("### Graph-Based Models and Feature-Based Baselines")

    col_left, col_right = st.columns([1.1, 1])
    with col_left:
        st.markdown("#### Graph Neural Network (GNN)")
        st.write(
            """
- Nodes: ~66k films from 2010–2025  
- Edges: shared genres, directors, production countries, and diversity similarity  
- Architecture: 2-layer GraphSAGE with batch norm, dropout, and residual connections  
- Goal: propagate information across similar films to improve audience score prediction.
            """
        )

        st.markdown("#### Knowledge Graph Convolutional Network (KGCN)")
        st.write(
            """
- Heterogeneous graph of movies, genres, directors, production companies, and countries  
- Relation-specific parameters to capture different semantic edge types  
- Learns rich embeddings for film entities that complement the GNN.
            """
        )

        st.markdown("#### XGBoost Baseline")
        st.write(
            """
- ~150 engineered features: metadata, engagement metrics, recency-weighted trailer stats, and diversity indicators  
- Gradient Boosting Regressor with tuned depth, learning rate, and number of trees  
- Strong individual performance and interpretable feature importances.
            """
        )

    with col_right:
        st.markdown("#### System Architecture")
        st.graphviz_chart(
            """
            digraph {
                rankdir=LR;
                TMDB -> Merge;
                RottenTomatoes -> Merge;
                YouTube -> Merge;
                Merge -> MongoDB;
                MongoDB -> "Feature Engineering";
                "Feature Engineering" -> GNN;
                "Feature Engineering" -> XGBoost;
                "Feature Engineering" -> KGCN;
                GNN -> Ensemble;
                KGCN -> Ensemble;
                XGBoost -> Ensemble;
                Ensemble -> "Audience Score Prediction";
            }
            """
        )

    st.markdown("---")
    st.markdown("### Ensemble Performance Summary")

    stacking_model = artifacts.get('stacking_model') if artifacts else None
    ensemble_meta = artifacts.get('ensemble_meta', {}) if artifacts else {}

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("GNN RMSE", "0.1929")
        st.caption("Graph-based similarity learning")
    with c2:
        st.metric("KGCN RMSE", "0.1709")
        st.caption("Knowledge graph relationships")
    with c3:
        st.metric("XGBoost RMSE", "0.1097")
        st.caption("Feature-engineered baseline")

    st.markdown("")
    c4, c5 = st.columns(2)
    with c4:
        if stacking_model is not None:
            rmse = ensemble_meta.get('stacking_rmse', 0.0996)
            st.metric("Stacking Ensemble RMSE", f"{rmse:.4f}")
        else:
            st.metric("Stacking Ensemble RMSE", "N/A")
    with c5:
        st.metric("Within ±10% of Audience Score", "≈80% (held-out set)")

    st.markdown("#### Original Weighted Average vs Stacking")
    comparison_data = {
        'Method': ['Original (Weighted Avg)', 'Stacking (Gradient Boosting)'],
        'RMSE': [0.1452, 0.0996],
        'MAE': [0.1170, 0.0669],
        'Within ±10%': ['48.9%', '80.2%'],
        'Within ±5%': ['24.2%', '54.5%']
    }
    st.dataframe(pd.DataFrame(comparison_data), hide_index=True, use_container_width=True)

def visual_graph_page(db):
    st.title("Visual Graph Explorer")

    st.write(
        """
Explore a local neighborhood of similar films in the graph.  
Select a movie to view its connections based on shared genres and popularity.
        """
    )

    titles = get_all_movie_titles(db)
    title = st.selectbox("Select a movie:", ["-- Select --"] + titles)
    if title != "-- Select --":
        movie = search_movie(db, title)
        if movie:
            fig = create_similarity_graph_figure(db, movie, max_neighbors=20)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough neighborhood information to build a similarity graph for this film.")
        else:
            st.error("Movie not found in database.")

def acknowledgements_page():
    st.title("Team & Acknowledgements")

    st.markdown("### Team 15 — Cinemaniacs")
    st.write(
        """
- Angela Cottone  
- Nidhi Deshmukh  
- Dylan Sidhu  
- Matthew Ward  
- Clara Wei  
        """
    )

    st.markdown("### Data Sources")
    st.write(
        """
- TMDB API  
- Rotten Tomatoes website  
- YouTube Data API  
        """
    )

    st.markdown("### Acknowledgements")
    st.write(
        """
We would like to thank the STA 160 instructional team for guidance and support throughout the project, 
as well as the maintainers of the external film data sources that made this work possible. 
We also acknowledge the use of AI-assisted tools for organization, prototyping, and documentation.
        """
    )

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    db = get_database_connection()
    if db is None:
        st.error("Unable to connect to database. Please check MongoDB credentials.")
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
            "Home",
            "Movie Search",
            "Compare Movies",
            "Analytics Dashboard",
            "Modeling",
            "Visual Graph Explorer",
            "Acknowledgements"
        ],
        key="page"
    )

    if page == "Home":
        home_page(db)
    elif page == "Movie Search":
        movie_search_page(db, artifacts)
    elif page == "Compare Movies":
        compare_movies_page(db, artifacts)
    elif page == "Analytics Dashboard":
        analytics_page(db)
    elif page == "Modeling":
        modeling_page(artifacts)
    elif page == "Visual Graph Explorer":
        visual_graph_page(db)
    elif page == "Acknowledgements":
        acknowledgements_page()

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Filmlytics | STA 160 Project | Team 15</p>
            <p>Predicting Movie Audience Scores Using Graph-Based Modeling</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
