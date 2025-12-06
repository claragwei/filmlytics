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
from pyvis.network import Network
import streamlit.components.v1 as components

# CONFIGURATION

# MongoDB Connection (from Streamlit secrets)
MONGODB_URI = st.secrets["MONGODB_URI"]

# Page Configuration
st.set_page_config(
    page_title="Cinemaniacs - Movie Success Prediction",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
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
    """Load all models, scalers, and prediction dataframes including stacking model."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    artifact_dir = os.path.join(script_dir, '.', 'model_artifacts')

    artifacts = {}

    required_files = {
        # XGBoost
        'xg_model': 'xgboost_base_model.pkl',
        'xg_features': 'xg_feature_columns.json',
        # GNN/KGCN shared
        'scaler': 'movie_feature_scaler_diversity.pkl',
        # GNN Predictions (Lookup)
        'gnn_preds': 'gnn_preds_all_movies.csv',
        # KGCN Predictions (Lookup)
        'kgcn_preds': 'kgcn_preds_all_movies.csv',
        # XGBoost Predictions (Lookup)
        'xgb_preds': 'xgb_preds_all_movies.csv',
    }

    for key, filename in required_files.items():
        path = os.path.join(artifact_dir, filename)
        if not os.path.exists(path):
            st.warning(f"Missing required file: {filename}. Prediction will be incomplete.")
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

    # Fallback ensemble weights
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
    """
    Placeholder for a feature generator matching the XGBoost training pipeline.
    Currently unused; predictions come from precomputed lookup tables.
    """
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

    predictions = {
        'gnn': gnn_pred,
        'kgcn': kgcn_pred,
        'xg': xgb_pred
    }

    stacking_model = artifacts.get('stacking_model')
    has_all_preds = not (np.isnan(gnn_pred) or np.isnan(kgcn_pred) or np.isnan(xgb_pred))

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

    valid_keys = valid_preds.keys()
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
        title='Distribution of Movie Ratings',
        labels={'value': 'Rating', 'count': 'Number of Movies'},
        color_discrete_sequence=['#1f77b4']
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
            title='Movie Success Rate Over Time',
            labels={'year': 'Year', 'success_rate': 'Success Rate (%)'},
            markers=True
        )
        fig.update_layout(hovermode='x unified')
        return fig
    return None

# =============================================================================
# PAGE FUNCTIONS
# =============================================================================

def introduction_page():
    st.title("Cinemaniacs: Predicting Film Audience Scores with Graph-Based Models")
    st.markdown("---")

    st.subheader("Abstract")
    st.write("""
        Cinemaniacs is a predictive analytics platform that forecasts Rotten Tomatoes audience scores 
        for upcoming films using a hybrid ensemble of Graph Neural Networks (GNN), 
        Knowledge-Graph Convolutional Networks (KGCN), and XGBoost. Our dataset spans over 66,000 films 
        from 2010–2025, integrating metadata from TMDB, Rotten Tomatoes, and YouTube trailers, including 
        engagement metrics, cast details, sentiment features, and gender representation.
    """)

    st.subheader("Motivation")
    st.write("""
        Audience scores influence marketing strategies, financial forecasting, and streaming platform decisions. 
        Traditional models rely solely on metadata and ignore the relational structure of the film ecosystem. 
        Our graph-based approach models collaborations, shared production patterns, and genre clusters, 
        revealing complex relationships that influence audience reception.
    """)

    st.subheader("Objectives")
    st.write("""
        - Build a unified movie dataset using TMDB, Rotten Tomatoes, and YouTube  
        - Engineer feature-rich representations across metadata, sentiment, and diversity  
        - Construct GNN, KGCN, and XGBoost models  
        - Develop an ensemble system for audience score prediction  
        - Build an interactive Streamlit dashboard  
    """)

    st.subheader("System Architecture")
    st.graphviz_chart("""
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
        XGBoost -> Ensemble;
        KCGN -> Ensemble;
        Ensemble -> "Audience Score Prediction";
    }
    """)

def home_page(db):
    st.title("Cinemaniacs")
    st.subheader("Ensemble Movie Success Prediction Platform")
    st.markdown("---")

    st.header("Welcome to Cinemaniacs")
        
    col1, col2, col3 = st.columns(3)
    stats = get_database_stats(db)
        
    with col1:
        st.metric("Total Movies", f"{stats['total']:,}")
    with col2:
        st.metric("Successful Movies", f"{stats['successful']:,}")
    with col3:
        st.metric("Success Rate", f"{(stats['successful'] / stats['total'] * 100):.1f}%")
        
    st.markdown("---")
        
    st.subheader("Top Rated Movies (TMDB)")
    top_movies = get_top_movies(db, limit=10)
        
    for i, movie in enumerate(top_movies[:5], 1):
        with st.expander(f"{i}. {movie['title']} - {movie['tmdb_metrics']['vote_average']}/10"):
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
                    st.write(f"**Overview:** {overview[:200]}...")

def data_pipeline_page():
    st.title("Data Pipeline Overview")
    st.markdown("---")

    st.subheader("Data Sources")
    st.write("""
        **TMDB (66,233 films)**  
        - Genres, runtime, cast, popularity, release information  
        - Budget, production metadata, posters, trailers  

        **Rotten Tomatoes (6,800 films)**  
        - Audience and critic scores  
        - Aggregated critic review sentiment  

        **YouTube API (42,156 trailers)**  
        - Trailer metadata and engagement metrics  
        - Upload timing, official trailer filtering, recency-based weighting  
    """)

    st.subheader("Cleaning and Integration")
    st.write("""
        - Merged datasets using film IDs, titles, and release years  
        - Removed duplicate trailers using heuristics  
        - Normalized numeric fields (log-transform where appropriate)  
        - Computed sentiment scores using DistilBERT  
        - Standardized gender representation features  
    """)

    st.subheader("MongoDB Schema")
    st.graphviz_chart("""
    digraph {
        rankdir=TB;
        Movie -> Production;
        Movie -> Metrics;
        Movie -> RottenTomatoes;
        Movie -> Trailer;
        Production -> Cast;
        Production -> Director;
    }
    """)

def modeling_overview_page():
    st.title("Modeling Framework")
    st.markdown("---")

    st.subheader("Graph Neural Network (GNN)")
    st.write("""
        The GNN treats each film as a node in a graph of over 66,000 movies.  
        Edges represent shared genres, directors, production countries, 
        and gender-representation similarity.  
        Architecture: Two-layer GraphSAGE with batch normalization, dropout, and residual connections.
    """)

    st.graphviz_chart("""
    digraph {
        rankdir=LR;
        Movie1 -> Movie2 [label="shared genre"];
        Movie1 -> Movie3 [label="director"];
        Movie1 -> Movie4 [label="diversity similarity"];
        MovieGraph -> GNNModel;
    }
    """)

    st.subheader("XGBoost Model")
    st.write("""
        - Over 150 engineered features  
        - Critic-score presence, runtime, production metadata  
        - Recency-weighted trailer engagement metrics  
        - Gender diversity features  
    """)

    st.subheader("Knowledge Graph Convolutional Network (KGCN)")
    st.write("""
        A relational model with over 300,000 nodes representing films, directors, genres, 
        production companies, and semantic relationships.  
        Learns relation-specific embeddings that capture structured metadata interactions.
    """)

    st.subheader("Model Performance Comparison")
    perf_df = pd.DataFrame({
        "Model": ["GNN", "KGCN", "XGBoost"],
        "RMSE": [0.197, 0.178, 0.177]
    }).set_index("Model")
    st.bar_chart(perf_df)

def movie_search_page(db, artifacts):
    st.title("Movie Search and Prediction")
    st.markdown("---")

    all_titles = get_all_movie_titles(db)

    selected_title = st.selectbox(
        "Select or type a movie title:",
        options=["-- Select a Movie --"] + all_titles,
        index=0
    )
        
    search_query = selected_title if selected_title != "-- Select a Movie --" else None
        
    if search_query:
        movie = search_movie(db, search_query) 
        if movie:
            st.success(f"Found: {movie['title']}")
                
            ensemble_score, breakdown = predict_ensemble(movie, artifacts)
                
            st.markdown("---")
            st.subheader("Ensemble Audience Score Prediction")
                
            if not np.isnan(ensemble_score):
                st.success(f"Predicted Audience Score: {ensemble_score*100:.1f}%")
                    
                st.markdown("#### Model Breakdown")
                b_col1, b_col2, b_col3 = st.columns(3)
                    
                with b_col1:
                    score = breakdown['gnn']
                    st.metric("GNN Prediction", f"{score*100:.1f}%" if not np.isnan(score) else "N/A")
                with b_col2:
                    score = breakdown['kgcn']
                    st.metric("KGCN Prediction", f"{score*100:.1f}%" if not np.isnan(score) else "N/A")
                with b_col3:
                    score = breakdown['xg']
                    st.metric("XGBoost Prediction", f"{score*100:.1f}%" if not np.isnan(score) else "N/A")
            else:
                st.warning("Cannot generate ensemble prediction (missing GNN/KGCN/XGB data for this movie).")
                
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
                    
                st.markdown("### Ratings")
                rating_col1, rating_col2, rating_col3 = st.columns(3)
                with rating_col1:
                    st.metric("TMDB Rating", f"{movie['tmdb_metrics'].get('vote_average', 'N/A')}/10")
                with rating_col2:
                    rt_crit = movie.get('rotten_tomatoes', {}).get('critic_score')
                    st.metric("RT Critics", rt_crit if rt_crit else "N/A")
                with rating_col3:
                    rt_aud = movie.get('rotten_tomatoes', {}).get('audience_score')
                    st.metric("RT Audience", rt_aud if rt_aud else "N/A")
                    
                if movie['tmdb_metrics'].get('is_successful'):
                    st.success("Classified as: SUCCESSFUL")
                else:
                    st.error("Classified as: NOT SUCCESSFUL")
                
            st.markdown("### Similar Movies")
            similar = get_similar_movies(db, movie['tmdb_id'], limit=5)
            if similar:
                cols = st.columns(5)
                for idx, sim_movie in enumerate(similar):
                    with cols[idx]:
                        st.write(f"**{sim_movie['title']}**")
                        st.write(f"{sim_movie['tmdb_metrics']['vote_average']}/10")
            else:
                st.write("No similar movies found")
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
            st.write(f"**{movie['title']}** - {movie['tmdb_metrics']['vote_average']}/10")

def analytics_page(db):
    st.title("Data Analytics Dashboard")
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
            st.info("Insufficient data for time series analysis")

def ensemble_model_page(artifacts):
    st.title("Stacking Ensemble Model")
    st.markdown("---")

    stacking_model = artifacts.get('stacking_model') if artifacts else None
    ensemble_meta = artifacts.get('ensemble_meta', {}) if artifacts else {}

    if stacking_model is not None:
        st.success("Stacking meta-learner loaded successfully.")
    else:
        st.warning("Stacking model not found. Using weighted average fallback in predictions.")

    st.write("""
        The platform uses a stacking ensemble that combines predictions from three base models
        to estimate Rotten Tomatoes audience scores (0–100%).

        Base models:
        - GNN (Graph Neural Network): Learns from the movie similarity graph  
        - KGCN (Knowledge Graph Convolutional Network): Uses genre/director/cast relationships  
        - XGBoost: Gradient boosting on a rich set of engineered features  
    """)

    st.markdown("---")

    if stacking_model is not None:
        st.subheader("Stacking Meta-Learner Performance")
        meta_cols = st.columns(3)
        with meta_cols[0]:
            model_name = ensemble_meta.get('meta_model', 'Gradient Boosting')
            st.metric("Meta-Model", model_name)
        with meta_cols[1]:
            rmse = ensemble_meta.get('stacking_rmse', 0.1085)
            st.metric("Stacking RMSE", f"{rmse:.4f}")
        with meta_cols[2]:
            st.metric("Accuracy (±10%)", "80.2%")

        st.info("""
            How stacking works:
            1. Each base model (GNN, KGCN, XGBoost) makes a prediction.  
            2. The meta-learner takes all three predictions as input.  
            3. It learns a non-linear combination of them.  
            4. It produces the final ensemble prediction.  
        """)

    st.markdown("---")
    st.subheader("Individual Model Performance")
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    with perf_col1:
        st.metric("GNN RMSE", "0.1929")
        st.caption("Graph-based similarity learning")
    with perf_col2:
        st.metric("KGCN RMSE", "0.1709")
        st.caption("Knowledge graph relationships")
    with perf_col3:
        st.metric("XGBoost RMSE", "0.1097")
        st.caption("Feature engineering (best individual)")

    st.markdown("---")
    st.subheader("Prediction Coverage")
    cov_col1, cov_col2, cov_col3 = st.columns(3)
    with cov_col1:
        gnn_preds_len = len(artifacts['gnn_preds']) if artifacts and artifacts.get('gnn_preds') is not None else 0
        st.metric("GNN Predictions", f"{gnn_preds_len:,}")
    with cov_col2:
        kgcn_preds_len = len(artifacts['kgcn_preds']) if artifacts and artifacts.get('kgcn_preds') is not None else 0
        st.metric("KGCN Predictions", f"{kgcn_preds_len:,}")
    with cov_col3:
        xgb_preds_len = len(artifacts['xgb_preds']) if artifacts and artifacts.get('xgb_preds') is not None else 0
        st.metric("XGBoost Predictions", f"{xgb_preds_len:,}")

    st.markdown("---")
    st.subheader("Original vs Stacking Comparison")

    comparison_data = {
        'Method': ['Original (Weighted Avg)', 'Stacking (Gradient Boosting)'],
        'RMSE': [0.1452, 0.0996],
        'MAE': [0.1170, 0.0669],
        'Within ±10%': ['48.9%', '80.2%'],
        'Within ±5%': ['24.2%', '54.5%']
    }
    st.dataframe(pd.DataFrame(comparison_data), hide_index=True, use_container_width=True)
    st.caption("Stacking achieves substantial RMSE improvement and predicts a much higher fraction of movies within a small error margin.")

def database_stats_page(db):
    st.title("Database Statistics")
    st.markdown("---")
        
    stats = get_database_stats(db)
        
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Movies", f"{stats['total']:,}")
    with col2:
        st.metric("Successful", f"{stats['successful']:,}")
    with col3:
        st.metric("With RT Data", f"{stats['with_rotten_tomatoes']:,}")
    with col4:
        st.metric("With Trailers", f"{stats['with_trailers']:,}")
        
    st.markdown("---")
        
    col1, col2 = st.columns(2)
        
    with col1:
        st.subheader("Data Completeness")
        completeness_data = {
            'Category': ['Rotten Tomatoes', 'Trailers', 'Success Labels'],
            'Percentage': [
                (stats['with_rotten_tomatoes'] / stats['total']) * 100,
                (stats['with_trailers'] / stats['total']) * 100,
                (stats['successful'] / stats['total']) * 100
            ]
        }
        df_complete = pd.DataFrame(completeness_data)
        fig = px.bar(
            df_complete,
            x='Category',
            y='Percentage',
            title='Data Completeness (%)',
            color='Percentage',
            color_continuous_scale='greens'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Collection Info")

        pipeline = [
            {"$match": {
                "release_info.tmdb_release_date": {
                    "$ne": None,
                    "$regex": "^[0-9]{4}",
                    "$type": "string"
                }
            }},
            {"$group": {
                "_id": None,
                "oldest_date": {"$min": "$release_info.tmdb_release_date"},
                "newest_date": {"$max": "$release_info.tmdb_release_date"}
            }}
        ]
        
        results = list(db.movies.aggregate(pipeline))
        
        if results:
            dates = results[0]
            oldest_movie = db.movies.find_one({"release_info.tmdb_release_date": dates["oldest_date"]})
            newest_movie = db.movies.find_one({"release_info.tmdb_release_date": dates["newest_date"]})
            
            if oldest_movie and newest_movie:
                st.write(f"**Oldest Movie:** {oldest_movie['title']} ({dates['oldest_date']})")
                st.write(f"**Newest Movie:** {newest_movie['title']} ({dates['newest_date']})")
            else:
                st.info("Could not retrieve movies for the calculated date range.")
        else:
            st.info("No valid release dates found in the database.")
            
        st.write("**Top 5 Genres:**")
        pipeline_genres = [
            {"$unwind": "$production.genres"},
            {"$group": {"_id": "$production.genres", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 5}
        ]
        for doc in db.movies.aggregate(pipeline_genres):
            st.write(f"- {doc['_id']}: {doc['count']:,}")

def compare_movies_page(db, artifacts):
    st.title("Compare Movies")
    st.markdown("---")

    titles = get_all_movie_titles(db)

    col1, col2 = st.columns(2)
    movie1 = col1.selectbox("Select first movie:", titles)
    movie2 = col2.selectbox("Select second movie:", titles)

    if movie1 and movie2:
        m1 = search_movie(db, movie1)
        m2 = search_movie(db, movie2)

        if m1 is None or m2 is None:
            st.error("One or both selected movies could not be found in the database.")
            return

        st.subheader("Basic Feature Comparison")
        comparison = pd.DataFrame({
            "Feature": ["TMDB Score", "Vote Count", "Runtime", "Budget"],
            movie1: [
                m1["tmdb_metrics"]["vote_average"],
                m1["tmdb_metrics"]["vote_count"],
                m1["production"].get("runtime", None),
                m1["production"].get("budget", None),
            ],
            movie2: [
                m2["tmdb_metrics"]["vote_average"],
                m2["tmdb_metrics"]["vote_count"],
                m2["production"].get("runtime", None),
                m2["production"].get("budget", None),
            ]
        })
        st.table(comparison)

        st.subheader("Prediction Comparison")
        pred1, _ = predict_ensemble(m1, artifacts)
        pred2, _ = predict_ensemble(m2, artifacts)

        c1, c2 = st.columns(2)
        c1.metric(movie1, f"{pred1*100:.1f}%" if not np.isnan(pred1) else "N/A")
        c2.metric(movie2, f"{pred2*100:.1f}%" if not np.isnan(pred2) else "N/A")

def interactive_graph_page(db):
    st.title("Film Similarity Graph")
    st.markdown("---")

    titles = get_all_movie_titles(db)
    choice = st.selectbox("Select a film to visualize connections:", titles)

    movie = search_movie(db, choice)
    if not movie:
        st.error("Selected movie not found in database.")
        return

    genres = movie["production"].get("genres", [])
    if not genres:
        st.info("Selected movie has no genre information; cannot construct similarity graph.")
        return

    # Use first genre for a simple local neighborhood
    genre = genres[0]
    neighbors = get_movies_by_genre(db, genre, limit=20)

    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
    net.add_node(movie["tmdb_id"], label=choice, color="gold")

    for m in neighbors:
        if m["tmdb_id"] == movie["tmdb_id"]:
            continue
        net.add_node(m["tmdb_id"], label=m["title"])
        net.add_edge(movie["tmdb_id"], m["tmdb_id"])

    net.save_graph("graph_temp.html")

    with open("graph_temp.html", "r", encoding="utf-8") as f:
        html = f.read()

    components.html(html, height=600)

def team_page():
    st.title("Team and Acknowledgments")
    st.markdown("---")

    st.write("""
        **Team 15 — Cinemaniacs Project**  
        - Clara Wei  
        - (Add remaining team members)
    """)

    st.write("""
        **Technologies Used**  
        - Streamlit  
        - MongoDB  
        - PyTorch Geometric  
        - XGBoost  
        - Hugging Face Transformers  
        - PyVis  
    """)

    st.write("""
        **Data Sources**  
        - TMDB API  
        - Rotten Tomatoes Website  
        - YouTube Data API  
    """)

    st.write("""
        **Acknowledgments**  
        Thanks to the STA 160 instructional team, and to AI-assisted tools used for 
        organization, prototyping, and documentation.
    """)

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
        st.sidebar.success("Artifacts loaded")

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
            "Ensemble Model",
            "Interactive Graph",
            "Team & Acknowledgments"
        ]
    )

    modeling_subpage = None
    if page == "Modeling":
        with st.sidebar.expander("Modeling & Data Views", expanded=True):
            modeling_subpage = st.radio(
                "Select view",
                ["Modeling Overview", "Data Pipeline", "Database Stats"],
                label_visibility="collapsed"
            )

    if page == "Introduction":
        introduction_page()
    elif page == "Home":
        home_page(db)
    elif page == "Modeling":
        if modeling_subpage == "Modeling Overview":
            modeling_overview_page()
        elif modeling_subpage == "Data Pipeline":
            data_pipeline_page()
        elif modeling_subpage == "Database Stats":
            database_stats_page(db)
    elif page == "Movie Search":
        movie_search_page(db, artifacts)
    elif page == "Compare Movies":
        compare_movies_page(db, artifacts)
    elif page == "Analytics":
        analytics_page(db)
    elif page == "Ensemble Model":
        ensemble_model_page(artifacts)
    elif page == "Interactive Graph":
        interactive_graph_page(db)
    elif page == "Team & Acknowledgments":
        team_page()

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Cinemaniacs | STA 160 Project | Team 15</p>
        <p>Ensemble Prediction Platform</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
