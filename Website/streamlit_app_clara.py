import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import certifi
import os
import joblib  # For loading XGBoost model and scaler
import json    # For loading feature list
import pickle  # For loading stacking model

# =============================================================================
# CONFIGURATION
# =============================================================================

# MongoDB Connection (from Streamlit secrets)
MONGODB_URI = st.secrets["MONGODB_URI"]

st.set_page_config(
    page_title="Cinemaniacs - Filmlytics",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global CSS: Blue Tech Theme
CUSTOM_CSS = """
<style>
/* Global font & background */
html, body, [class*="css"]  {
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Hero section */
.hero-container {
    background: linear-gradient(120deg, #020617, #0f172a 40%, #1e293b 80%);
    padding: 3.5rem 3rem;
    border-radius: 18px;
    color: #e5e7eb;
    margin-bottom: 2rem;
    border: 1px solid rgba(148,163,184,0.4);
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
    color: #f9fafb;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #cbd5f5;
    max-width: 700px;
}

.hero-badge {
    display: inline-block;
    padding: 0.25rem 0.7rem;
    font-size: 0.75rem;
    border-radius: 999px;
    background: rgba(15,23,42,0.7);
    border: 1px solid rgba(148,163,184,0.6);
    color: #e5e7eb;
    margin-bottom: 0.75rem;
}

.hero-buttons {
    margin-top: 1.5rem;
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
}

/* Primary button style override */
.stButton>button.primary-btn {
    background: #2563EB;
    color: #f9fafb;
    border-radius: 999px;
    border: none;
    padding: 0.5rem 1.3rem;
    font-size: 0.95rem;
    font-weight: 500;
    cursor: pointer;
}
.stButton>button.primary-btn:hover {
    background: #1d4ed8;
}

/* Secondary button */
.stButton>button.secondary-btn {
    background: transparent;
    color: #e5e7eb;
    border-radius: 999px;
    border: 1px solid rgba(148,163,184,0.7);
    padding: 0.5rem 1.3rem;
    font-size: 0.95rem;
    font-weight: 500;
    cursor: pointer;
}
.stButton>button.secondary-btn:hover {
    background: rgba(15,23,42,0.85);
}

/* Metric cards on hero */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 0.8rem;
    margin-top: 2rem;
}

.metric-card {
    background: rgba(15,23,42,0.9);
    border-radius: 14px;
    padding: 0.8rem 1rem;
    border: 1px solid rgba(51,65,85,0.9);
}

.metric-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #9ca3af;
}

.metric-value {
    font-size: 1.15rem;
    font-weight: 600;
    color: #e5e7eb;
}

/* Section titles */
.section-title {
    font-size: 1.25rem;
    font-weight: 600;
    margin-top: 1.5rem;
    margin-bottom: 0.3rem;
}

/* Info cards */
.info-card {
    background: #f9fafb;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    border: 1px solid #e5e7eb;
    margin-bottom: 0.75rem;
}

/* Make dataframes a bit cleaner */
.dataframe tbody tr:nth-child(even) {
    background-color: #f9fafb;
}

/* Footer */
.app-footer {
    text-align: center;
    font-size: 0.85rem;
    color: #6b7280;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #e5e7eb;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

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
        db.movies.count_documents({})  # Test connection
        return db
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

# =============================================================================
# ARTIFACT LOADING AND ENSEMBLE PREDICTION SETUP
# =============================================================================

@st.cache_resource
def load_ensemble_artifacts():
    """Load all models, scalers, and prediction dataframes including stacking model."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    artifact_dir = os.path.join(script_dir, '.', 'model_artifacts')

    artifacts = {}

    required_files = {
        'xg_model': 'xgboost_base_model.pkl',
        'xg_features': 'xg_feature_columns.json',
        'scaler': 'movie_feature_scaler_diversity.pkl',
        'gnn_preds': 'gnn_preds_all_movies.csv',
        'kgcn_preds': 'kgcn_preds_all_movies.csv',
        'xgb_preds': 'xgb_preds_all_movies.csv',
    }

    for key, filename in required_files.items():
        path = os.path.join(artifact_dir, filename)
        if not os.path.exists(path):
            st.warning(f"Missing required file: {filename}. Prediction will be partially available.")
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
                else:
                    artifacts[key] = df.set_index('tmdb_id').iloc[:, 0]
        except Exception as e:
            return None, f"Error loading {filename}: {e}"

    # Stacking meta-model
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

    artifacts['ensemble_weights'] = {'gnn': 0.33, 'kgcn': 0.34, 'xg': 0.33}
    return artifacts, None

# Placeholder for potential future XGBoost live features
def parse_pct_string(s):
    if isinstance(s, str) and s.endswith("%"):
        try:
            return float(s[:-1].strip()) / 100.0
        except Exception:
            return np.nan
    return np.nan

def generate_xgboost_features(movie_data, artifacts):
    """Placeholder for feature generator matching the XGBoost training pipeline."""
    return None

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

    predictions = {'gnn': gnn_pred, 'kgcn': kgcn_pred, 'xg': xgb_pred}

    stacking_model = artifacts.get('stacking_model')
    has_all = not (np.isnan(gnn_pred) or np.isnan(kgcn_pred) or np.isnan(xgb_pred))

    if stacking_model is not None and has_all:
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
    total_weight = sum(weights.get(k, 0) for k in valid_keys)

    if total_weight == 0:
        ensemble_pred = np.mean(list(valid_preds.values()))
    else:
        ensemble_pred = sum(valid_preds[k] * (weights[k] / total_weight) for k in valid_keys)

    ensemble_pred = np.clip(ensemble_pred, 0.0, 1.0)
    return ensemble_pred, predictions

# =============================================================================
# DATA QUERY FUNCTIONS
# =============================================================================

@st.cache_data
def get_all_movie_titles(_db):
    """Get a list of UNIQUE movie titles, sorted by vote count."""
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
# VISUALIZATION FUNCTIONS
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
        df = df[df['year'].astype(int) >= 2000]
        if df.empty:
            return None
        
        fig = px.line(
            df,
            x='year',
            y='success_rate',
            title='Movie Success Rate Over Time (TMDB-based)',
            labels={'year': 'Year', 'success_rate': 'Success Rate (%)'},
            markers=True
        )
        fig.update_layout(hovermode='x unified')
        return fig
    return None

# =============================================================================
# PAGE FUNCTIONS
# =============================================================================

def introduction_page(db, artifacts):
    stats = get_database_stats(db)

    total_movies = stats['total']
    labeled_movies = stats['successful']  # not exactly labels but a proxy for “has outcome”
    with_trailers = stats['with_trailers']

    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-badge">STA 160 · Team 15 · Cinemaniacs</div>
            <div class="hero-title">Filmlytics: Predicting Film Audience Scores with Graph-Based Models</div>
            <div class="hero-subtitle">
                A full-stack analytics platform that forecasts Rotten Tomatoes audience scores 
                for thousands of films using Graph Neural Networks, Knowledge Graph models, 
                and gradient-boosted trees built on a unified movie knowledge base.
            </div>
        """,
        unsafe_allow_html=True
    )

    # Hero buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Explore Film Dashboard", key="hero_dashboard", type="primary"):
            st.session_state["nav_page"] = "Film Explorer"
            st.experimental_rerun()
    with col2:
        if st.button("View Modeling & Data", key="hero_modeling"):
            st.session_state["nav_page"] = "Modeling & Data"
            st.experimental_rerun()

    # Metric cards
    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Movies in Graph</div>
                <div class="metric-value">{total_movies:,}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Films with Audience Scores</div>
                <div class="metric-value">≈ 5,000</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Trailers with Engagement Data</div>
                <div class="metric-value">{with_trailers:,}</div>
            </div>
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("")
    st.markdown("### Motivation")
    st.write(
        """
        Audience scores shape marketing decisions, streaming acquisitions, and long-term film value. 
        Yet, metadata, engagement signals, and diversity information are fragmented across platforms 
        like TMDB, Rotten Tomatoes, and YouTube. Our goal is to unify these sources and build 
        models that not only predict audience scores, but also reveal how film attributes, 
        trailer engagement, and gender representation interact.
        """
    )

    st.markdown("### Objectives")
    st.write(
        """
        - Construct a unified film dataset (2010–2025) integrating TMDB, Rotten Tomatoes, and YouTube trailers.  
        - Engineer feature-rich representations across production metadata, sentiment, engagement, and gender representation.  
        - Train GNN, KGCN, and XGBoost models to predict Rotten Tomatoes audience scores.  
        - Build an ensemble that combines feature-level and relational information.  
        - Deliver an interactive dashboard for exploring predictions, model behavior, and film relationships.  
        """
    )

    st.markdown("### System Architecture")
    st.graphviz_chart(
        """
        digraph {
            rankdir=LR;
            node [shape=box, style="rounded,filled", color="#0f172a", fontcolor="#0f172a", fillcolor="#e5e7eb"];
            edge [color="#4b5563"];

            TMDB [label="TMDB\n(66k+ films)"];
            RT [label="Rotten Tomatoes\n(audience & critic scores)"];
            YT [label="YouTube API\n(trailers & engagement)"];
            Merge [label="Data Cleaning\n & Integration"];
            Mongo [label="MongoDB\nFilm Knowledge Base"];
            FE [label="Feature Engineering"];
            GNN [label="GNN"];
            KGCN [label="KGCN"];
            XGB [label="XGBoost"];
            Ensemble [label="Stacking Ensemble"];
            Dashboard [label="Filmlytics Dashboard"];

            TMDB -> Merge;
            RT -> Merge;
            YT -> Merge;
            Merge -> Mongo;
            Mongo -> FE;
            FE -> GNN;
            FE -> KGCN;
            FE -> XGB;
            GNN -> Ensemble;
            KGCN -> Ensemble;
            XGB -> Ensemble;
            Ensemble -> Dashboard;
        }
        """
    )

    st.markdown("### Key Outcomes")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="info-card">
            <strong>Graph Neural Network</strong><br/>
            Captures film-to-film similarity via genres, directors, production countries, 
            and gender representation; achieves meaningful correlation with audience scores.
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <div class="info-card">
            <strong>XGBoost Model</strong><br/>
            Uses 150+ engineered features including trailer engagement and diversity signals, 
            achieving the strongest point prediction accuracy among individual models.
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            """
            <div class="info-card">
            <strong>Ensemble Stacking</strong><br/>
            Combines GNN, KGCN, and XGBoost predictions to improve RMSE and the fraction of films 
            predicted within a ±10% error band.
            </div>
            """,
            unsafe_allow_html=True
        )

def modeling_overview_tab():
    st.subheader("Modeling Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="info-card">
            <strong>Graph Neural Network (GNN)</strong><br/>
            - Nodes: 66k films<br/>
            - Edges: shared genres, directors, countries, gender-similar films<br/>
            - Architecture: 2-layer GraphSAGE + residual connection<br/>
            - Semi-supervised: unlabeled films participate via message passing.
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <div class="info-card">
            <strong>Knowledge Graph Convolutional Network (KGCN)</strong><br/>
            - Nodes: films, people, genres, production entities<br/>
            - Relation-aware message passing across a heterogeneous graph<br/>
            - Learns which relation types are most predictive of audience scores.
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            """
            <div class="info-card">
            <strong>XGBoost</strong><br/>
            - 150+ engineered features<br/>
            - Production metadata, critic info, sentiment, engagement, gender metrics<br/>
            - Hyperparameters tuned via randomized search and cross validation.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("#### Ensemble Strategy")
    st.write(
        """
        We train each base model independently, then use a stacking meta-learner to combine 
        their predictions. When the meta-learner is unavailable, we fall back to a weighted 
        average of the three base models. This allows us to blend structural information 
        from the graph with high-resolution tabular features.
        """
    )

    perf_df = pd.DataFrame(
        {
            "Model": ["GNN", "KGCN", "XGBoost"],
            "RMSE": [0.197, 0.178, 0.177]
        }
    )
    fig = px.bar(
        perf_df,
        x="Model",
        y="RMSE",
        title="Model Performance (RMSE, lower is better)",
        text="RMSE"
    )
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(yaxis=dict(range=[0, 0.25]))
    st.plotly_chart(fig, use_container_width=True)

def data_pipeline_tab():
    st.subheader("Data Pipeline")

    st.markdown(
        """
        <div class="info-card">
        <strong>Data Sources</strong><br/>
        <ul>
        <li><b>TMDB (66,233 films)</b>: genres, runtime, budget, production metadata, cast, directors, posters, trailers, and TMDB vote statistics.</li>
        <li><b>Rotten Tomatoes (~6,800 films)</b>: audience scores, critic scores, and aggregate review sentiment.</li>
        <li><b>YouTube API (~42,000 trailers)</b>: trailer metadata, view/like/comment counts, timing relative to release, and engagement velocity.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-card">
        <strong>Cleaning & Integration</strong><br/>
        <ul>
        <li>Merge films by TMDB IDs or title + release year alignment.</li>
        <li>Filter out low-signal films (e.g., zero votes; unreliable scores).</li>
        <li>Deduplicate trailers and prioritize official, earliest trailers.</li>
        <li>Log-transform and normalize skewed engagement features.</li>
        <li>Compute sentiment scores for descriptions and critic text using DistilBERT.</li>
        <li>Standardize gender representation metrics from TMDB cast information.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("#### MongoDB Schema (High-Level)")
    st.graphviz_chart(
        """
        digraph {
            rankdir=TB;
            node [shape=box, style="rounded,filled", color="#0f172a", fontcolor="#0f172a", fillcolor="#e5e7eb"];
            edge [color="#4b5563"];

            Movie [label="Movie Document"];
            Production [label="production"];
            Metrics [label="tmdb_metrics"];
            RT [label="rotten_tomatoes"];
            Trailer [label="trailer"];
            People [label="cast / crew"];

            Movie -> Production;
            Movie -> Metrics;
            Movie -> RT;
            Movie -> Trailer;
            Production -> People;
        }
        """
    )

def database_stats_tab(db):
    st.subheader("Database Stats")

    stats = get_database_stats(db)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Movies", f"{stats['total']:,}")
    with col2:
        st.metric("Successful (vote_avg > 6)", f"{stats['successful']:,}")
    with col3:
        st.metric("With RT Data", f"{stats['with_rotten_tomatoes']:,}")
    with col4:
        st.metric("With Trailers", f"{stats['with_trailers']:,}")

    st.markdown("")

    c1, c2 = st.columns(2)
    with c1:
        completeness_data = {
            'Category': ['Rotten Tomatoes', 'Trailers', 'Success Labels'],
            'Percentage': [
                (stats['with_rotten_tomatoes'] / stats['total']) * 100 if stats['total'] else 0,
                (stats['with_trailers'] / stats['total']) * 100 if stats['total'] else 0,
                (stats['successful'] / stats['total']) * 100 if stats['total'] else 0
            ]
        }
        df_complete = pd.DataFrame(completeness_data)
        fig = px.bar(
            df_complete,
            x='Category',
            y='Percentage',
            title='Data Completeness (%)',
            color='Percentage',
            color_continuous_scale='blues'
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Top 5 Genres in Collection**")
        pipeline_genres = [
            {"$unwind": "$production.genres"},
            {"$group": {"_id": "$production.genres", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 5}
        ]
        for doc in db.movies.aggregate(pipeline_genres):
            st.write(f"- {doc['_id']}: {doc['count']:,} films")

def visual_analytics_page(db):
    st.title("Visual Analytics")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Genre Analysis", "Rating Distribution", "Success Trends"])

    with tab1:
        st.subheader("Genre Distribution")
        fig = create_genre_distribution_chart(db)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Rating Distribution")
        fig = create_rating_distribution(db)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Success Rate Over Time")
        fig = create_success_over_time(db)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data for time series analysis.")

def film_explorer_page(db, artifacts):
    st.title("Film Explorer")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Search", "Compare", "Recommendations"])

    all_titles = get_all_movie_titles(db)

    # --- SEARCH TAB ---
    with tab1:
        st.subheader("Search and Predict")

        selected_title = st.selectbox(
            "Select or type a movie title:",
            options=["-- Select a Movie --"] + all_titles,
            index=0,
            key="search_select"
        )

        search_query = selected_title if selected_title != "-- Select a Movie --" else None
        
        if search_query:
            movie = search_movie(db, search_query)
            if movie:
                ensemble_score, breakdown = predict_ensemble(movie, artifacts)

                upper = st.container()
                with upper:
                    col1, col2 = st.columns([1.2, 2])
                    with col1:
                        poster = movie.get('content', {}).get('poster_url')
                        if poster:
                            st.image(poster, width=230)
                        st.markdown(f"**TMDB ID:** {movie['tmdb_id']}")
                        st.markdown(f"**Genres:** {', '.join(movie['production'].get('genres', [])) or 'N/A'}")
                        st.markdown(f"**Runtime:** {movie['production'].get('runtime', 'N/A')} minutes")
                        st.markdown(f"**Budget:** ${movie['production'].get('budget', 0):,.0f}")
                        st.markdown(f"**Release Date:** {movie['release_info'].get('tmdb_release_date', 'N/A')}")

                    with col2:
                        st.subheader(movie['title'])

                        st.markdown("**Ratings**")
                        r1, r2, r3 = st.columns(3)
                        with r1:
                            st.metric("TMDB Rating", f"{movie['tmdb_metrics'].get('vote_average', 'N/A')}/10")
                        with r2:
                            rt_crit = movie.get('rotten_tomatoes', {}).get('critic_score')
                            st.metric("RT Critics", rt_crit if rt_crit else "N/A")
                        with r3:
                            rt_aud = movie.get('rotten_tomatoes', {}).get('audience_score')
                            st.metric("RT Audience", rt_aud if rt_aud else "N/A")

                        if movie['tmdb_metrics'].get('is_successful'):
                            st.success("Classified as: SUCCESSFUL (vote_average > 6)")
                        else:
                            st.info("Classified as: NOT SUCCESSFUL (vote_average ≤ 6)")

                        st.markdown("---")
                        st.markdown("**Ensemble Audience Score Prediction**")
                        if not np.isnan(ensemble_score):
                            st.metric("Predicted Audience Score", f"{ensemble_score*100:.1f}%")
                            b1, b2, b3 = st.columns(3)
                            with b1:
                                score = breakdown['gnn']
                                st.metric("GNN", f"{score*100:.1f}%" if not np.isnan(score) else "N/A")
                            with b2:
                                score = breakdown['kgcn']
                                st.metric("KGCN", f"{score*100:.1f}%" if not np.isnan(score) else "N/A")
                            with b3:
                                score = breakdown['xg']
                                st.metric("XGBoost", f"{score*100:.1f}%" if not np.isnan(score) else "N/A")
                        else:
                            st.warning("Cannot generate ensemble prediction for this film.")

                st.markdown("### Similar Films (Genre-Based)")
                similar = get_similar_movies(db, movie['tmdb_id'], limit=6)
                if similar:
                    cols = st.columns(3)
                    for idx, sim_movie in enumerate(similar):
                        with cols[idx % 3]:
                            st.markdown(f"**{sim_movie['title']}**")
                            st.caption(f"TMDB: {sim_movie['tmdb_metrics']['vote_average']:.1f}/10")
                else:
                    st.write("No similar movies found.")
            else:
                st.error(f"No movie found matching '{search_query}'")

    # --- COMPARE TAB ---
    with tab2:
        st.subheader("Compare Two Films")

        c1, c2 = st.columns(2)
        m1_title = c1.selectbox("First film", all_titles, key="cmp1")
        m2_title = c2.selectbox("Second film", all_titles, key="cmp2")

        if m1_title and m2_title:
            m1 = search_movie(db, m1_title)
            m2 = search_movie(db, m2_title)

            if m1 is None or m2 is None:
                st.error("One or both selected movies could not be found in the database.")
            else:
                # Basic metrics table
                st.markdown("#### Basic Features")
                comp_df = pd.DataFrame({
                    "Feature": ["TMDB Score", "Vote Count", "Runtime (min)", "Budget ($)"],
                    m1_title: [
                        m1["tmdb_metrics"]["vote_average"],
                        m1["tmdb_metrics"]["vote_count"],
                        m1["production"].get("runtime", None),
                        m1["production"].get("budget", None),
                    ],
                    m2_title: [
                        m2["tmdb_metrics"]["vote_average"],
                        m2["tmdb_metrics"]["vote_count"],
                        m2["production"].get("runtime", None),
                        m2["production"].get("budget", None),
                    ]
                })
                st.table(comp_df)

                st.markdown("#### Ensemble Prediction Comparison")
                pred1, _ = predict_ensemble(m1, artifacts)
                pred2, _ = predict_ensemble(m2, artifacts)

                mc1, mc2 = st.columns(2)
                mc1.metric(m1_title, f"{pred1*100:.1f}%" if not np.isnan(pred1) else "N/A")
                mc2.metric(m2_title, f"{pred2*100:.1f}%" if not np.isnan(pred2) else "N/A")

                # Radar chart for visual comparison (normalized)
                st.markdown("#### Feature Radar (Normalized)")

                def extract_numeric_features(movie, pred):
                    return {
                        "TMDB Score": movie["tmdb_metrics"]["vote_average"] or 0,
                        "Predicted Audience %": (pred * 100) if not np.isnan(pred) else 0,
                        "Vote Count (log10)": np.log10(movie["tmdb_metrics"]["vote_count"] + 1),
                        "Runtime (min)": movie["production"].get("runtime", 0) or 0,
                        "Budget (log10)": np.log10((movie["production"].get("budget", 0) or 0) + 1)
                    }

                f1 = extract_numeric_features(m1, pred1)
                f2 = extract_numeric_features(m2, pred2)

                categories = list(f1.keys())
                v1 = np.array(list(f1.values()), dtype=float)
                v2 = np.array(list(f2.values()), dtype=float)

                all_vals = np.concatenate([v1, v2])
                if all_vals.max() > all_vals.min():
                    v1_norm = (v1 - all_vals.min()) / (all_vals.max() - all_vals.min())
                    v2_norm = (v2 - all_vals.min()) / (all_vals.max() - all_vals.min())
                else:
                    v1_norm = np.zeros_like(v1)
                    v2_norm = np.zeros_like(v2)

                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=v1_norm.tolist() + [v1_norm[0]],
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=m1_title
                ))
                fig.add_trace(go.Scatterpolar(
                    r=v2_norm.tolist() + [v2_norm[0]],
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=m2_title
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

    # --- RECOMMENDATIONS TAB ---
    with tab3:
        st.subheader("Genre-Based Recommendations")

        base_title = st.selectbox(
            "Choose a reference film",
            options=all_titles,
            key="reco_base"
        )
        if base_title:
            base_movie = search_movie(db, base_title)
            if base_movie:
                st.markdown(f"Recommendations based on **{base_movie['title']}** and shared genres.")
                recs = get_similar_movies(db, base_movie['tmdb_id'], limit=9)
                if recs:
                    cols = st.columns(3)
                    for i, rec in enumerate(recs):
                        with cols[i % 3]:
                            poster = rec.get('content', {}).get('poster_url')
                            if poster:
                                st.image(poster, use_column_width=True)
                            st.markdown(f"**{rec['title']}**")
                            st.caption(f"TMDB: {rec['tmdb_metrics']['vote_average']:.1f}/10")
                else:
                    st.info("No recommendations found. Try another film.")
            else:
                st.error("Reference film could not be found in the database.")

def ensemble_model_page(artifacts):
    st.title("Ensemble Prediction Model")
    st.markdown("---")

    stacking_model = artifacts.get('stacking_model') if artifacts else None
    ensemble_meta = artifacts.get('ensemble_meta', {}) if artifacts else {}

    if stacking_model is not None:
        st.success("Stacking meta-learner loaded successfully.")
    else:
        st.info("Stacking model not available. The app falls back to a weighted average of GNN, KGCN, and XGBoost predictions.")

    st.markdown("### Ensemble Architecture")
    st.graphviz_chart(
        """
        digraph {
            rankdir=LR;
            node [shape=box, style="rounded,filled", color="#0f172a", fontcolor="#0f172a", fillcolor="#e5e7eb"];
            edge [color="#4b5563"];

            GNN [label="GNN Prediction"];
            KGCN [label="KGCN Prediction"];
            XGB [label="XGBoost Prediction"];
            Meta [label="Meta-Learner\n(Stacking)"];
            Final [label="Final Audience Score"];

            GNN -> Meta;
            KGCN -> Meta;
            XGB -> Meta;
            Meta -> Final;
        }
        """
    )

    st.markdown("### Performance Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        model_name = ensemble_meta.get('meta_model', 'Gradient Boosting')
        st.metric("Meta-Model", model_name)
    with col2:
        rmse = ensemble_meta.get('stacking_rmse', 0.1085)
        st.metric("Stacking RMSE", f"{rmse:.4f}")
    with col3:
        st.metric("Accuracy (±10%)", "80.2%")

    st.markdown(
        """
        <div class="info-card">
        <strong>How Stacking Works</strong><br/>
        Each base model is trained on the same set of films and produces its own prediction. 
        The stacking meta-learner is then trained on these predictions (and optionally 
        additional features) to learn an optimal non-linear combination. This typically 
        improves robustness and accuracy over any single model.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Original vs Stacking")
    comparison_data = {
        'Method': ['Original (Weighted Avg)', 'Stacking (Gradient Boosting)'],
        'RMSE': [0.1452, 0.0996],
        'MAE': [0.1170, 0.0669],
        'Within ±10%': ['48.9%', '80.2%'],
        'Within ±5%': ['24.2%', '54.5%']
    }
    st.dataframe(pd.DataFrame(comparison_data), hide_index=True, use_container_width=True)

def team_page():
    st.title("Team & Acknowledgments")
    st.markdown("---")

    st.markdown(
        """
        <div class="info-card">
        <strong>Team 15 — Cinemaniacs</strong><br/>
        Angela Cottone · Nidhi Deshmukh · Dylan Sidhu · Matthew Ward · Clara Wei
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-card">
        <strong>Technologies</strong><br/>
        Streamlit · MongoDB · PyTorch Geometric · XGBoost · Hugging Face Transformers · Python
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-card">
        <strong>Data Sources</strong><br/>
        TMDB API · Rotten Tomatoes Website · YouTube Data API
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-card">
        <strong>Acknowledgments</strong><br/>
        Thanks to the STA 160 instructional team, and to AI-assisted tools (ChatGPT, Google Gemini, GitHub Copilot) 
        used for prototyping, debugging, and documentation support.
        </div>
        """,
        unsafe_allow_html=True
    )

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    db = get_database_connection()
    if db is None:
        st.error("Unable to connect to database. Please check your MongoDB credentials.")
        st.stop()
    
    artifacts, error = load_ensemble_artifacts()
    if error:
        st.error(f"Ensemble artifacts failed to load: {error}")
        st.stop()
    else:
        st.sidebar.success("Model artifacts loaded")

    # Sidebar navigation with session_state-controlled page
    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = "Introduction"

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Introduction",
            "Modeling & Data",
            "Visual Analytics",
            "Film Explorer",
            "Ensemble Model",
            "Team & Acknowledgments"
        ],
        key="nav_page"
    )

    if page == "Introduction":
        introduction_page(db, artifacts)
    elif page == "Modeling & Data":
        st.title("Modeling & Data")
        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["Modeling Overview", "Data Pipeline", "Database Stats"])
        with tab1:
            modeling_overview_tab()
        with tab2:
            data_pipeline_tab()
        with tab3:
            database_stats_tab(db)
    elif page == "Visual Analytics":
        visual_analytics_page(db)
    elif page == "Film Explorer":
        film_explorer_page(db, artifacts)
    elif page == "Ensemble Model":
        ensemble_model_page(artifacts)
    elif page == "Team & Acknowledgments":
        team_page()

    st.markdown(
        """
        <div class="app-footer">
            Filmlytics · STA 160 · Team 15
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
