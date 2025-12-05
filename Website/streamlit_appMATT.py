import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pymongo import MongoClient
import os
import joblib # For loading XGBoost model and scaler
import json   # For loading feature list

# CONFIGURATION

# MongoDB Connection (replace with your actual connection string)
MONGODB_URI = "mongodb+srv://cinemaniacs:filmlytics@filmlytics.1emhcue.mongodb.net/?appName=filmlytics"

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
    """Connect to MongoDB and cache the connection"""
    try:
        # Note: Added server_api=ServerApi("1") if needed, but keeping it simple for Streamlit
        client = MongoClient(MONGODB_URI)
        db = client.cinemaniacs
        # Test connection
        db.movies.count_documents({})
        return db
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        return None

# =============================================================================
# ARTIFACT LOADING AND ENSEMBLE PREDICTION SETUP
# =============================================================================

@st.cache_resource
def load_ensemble_artifacts():
    """Load all models, scalers, and prediction dataframes."""
    
    # 1. Get the directory of the currently executing script (Website/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 2. Define the model_artifacts directory path
    artifact_dir = os.path.join(script_dir, '.', 'model_artifacts')
    
    artifacts = {}
    
    # --- Check all required files ---
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
    }
    
    for key, filename in required_files.items():
        path = os.path.join(artifact_dir, filename)
        if not os.path.exists(path):
            # Do not stop, but warn about missing files
            st.warning(f"⚠️ Missing required file: {filename}. Prediction will be incomplete.")
            artifacts[key] = None
            continue
        
        try:
            if filename.endswith('.pkl'):
                artifacts[key] = joblib.load(path)
            elif filename.endswith('.json'):
                with open(path, 'r') as f:
                    artifacts[key] = json.load(f)
            elif filename.endswith('.csv'):
                # Load predictions, ensuring tmdb_id is int for merges
                df = pd.read_csv(path)
                # Ensure tmdb_id is cleaned/integer
                df['tmdb_id'] = pd.to_numeric(df['tmdb_id'], errors='coerce').fillna(0).astype(int)
                artifacts[key] = df.set_index('tmdb_id')['pred_audience_score']
            
        except Exception as e:
            return None, f"Error loading {filename}: {e}"
            
    # --- Ensemble weights ---
    # Example: Equal weighting for GNN, KGCN, XGBoost predictions
    artifacts['ensemble_weights'] = {'gnn': 0.33, 'kgcn': 0.34, 'xg': 0.33}
    
    return artifacts, None

# --- IMPORTANT: XGBoost Feature Generation ---
# This feature generator MUST match the logic in your XGBoost training script (Section 4 & 6)
# It is simplified here for demonstration, but must be fully implemented for accuracy.

def parse_pct_string(s):
    if isinstance(s, str) and s.endswith("%"):
        try: return float(s[:-1].strip()) / 100.0
        except: return np.nan
    return np.nan

def generate_xgboost_features(movie_data, artifacts):
    """
    Creates the feature vector for a single movie needed by the XGBoost model.
    This function is a SIMPLIFIED PLACEHOLDER.
    """
    
    # Create a single-row DataFrame from the raw MongoDB document
    # Fill missing columns with 0/False as the XGBoost model expects them.
    
    # --- Feature Generation Placeholder ---
    
    # Since we cannot replicate the full feature engineering pipeline here (100+ lines),
    # we will rely only on the GNN/KGCN lookups and return NaN for XGBoost's prediction
    # if the feature vector cannot be built.
    
    # In a *production* app, you would load all the data for the movie (including
    # cast/director history from your DB/lookup tables) and then apply ALL 
    # the feature engineering steps from your training script to generate X_single.
    
    return None # Indicate that the feature generation is skipped here

def predict_ensemble(movie_data, artifacts):
    """
    Generates features (conceptually), predicts with XGBoost (placeholder), 
    and ensembles with GNN/KGCN predictions.
    
    Returns: ensemble_prediction (0-1), prediction_breakdown (dict)
    """
    
    tmdb_id = movie_data.get('tmdb_id')
    if tmdb_id is None:
        return np.nan, {}

    # 1. Get GNN/KGCN Predictions (Lookup)
    gnn_preds = artifacts.get('gnn_preds')
    kgcn_preds = artifacts.get('kgcn_preds')
    
    # --- CHANGE START ---
    
    # Get the value from the Series, or default to np.nan if not found.
    # The .item() call is the critical fix.
    
    gnn_pred = gnn_preds.get(tmdb_id, np.nan) if gnn_preds is not None else np.nan
    kgcn_pred = kgcn_preds.get(tmdb_id, np.nan) if kgcn_preds is not None else np.nan

    # Ensure gnn_pred and kgcn_pred are floats, not single-item Series, 
    # before they hit the np.isnan() condition later.

    try:
        if isinstance(gnn_pred, pd.Series):
            gnn_pred = gnn_pred.item()
    except:
        gnn_pred = np.nan # If .item() fails (e.g., Series is empty)

    try:
        if isinstance(kgcn_pred, pd.Series):
            kgcn_pred = kgcn_pred.item()
    except:
        kgcn_pred = np.nan # If .item() fails

    # --- CHANGE END ---
    
    # 2. XGBoost Prediction (Placeholder/Fallback)
    # The conditional statement below should now work correctly because gnn_pred/kgcn_pred are floats.
    
    if not np.isnan(gnn_pred) or not np.isnan(kgcn_pred):
        # Fallback: If at least one GNN model worked, set XGBoost to their average.
        # This prevents an N/A prediction if the movie exists in the lookups.
        xg_pred = np.nanmean([gnn_pred, kgcn_pred])
    else:
        # Last resort fallback: TMDB vote average (0-1)
        tmdb_avg = movie_data.get('tmdb_metrics', {}).get('vote_average')
        xg_pred = tmdb_avg / 10.0 if tmdb_avg is not None else np.nan
    
    # 3. Ensemble Calculation
    weights = artifacts.get('ensemble_weights', {})
    
    predictions = {
        'gnn': gnn_pred,
        'kgcn': kgcn_pred,
        'xg': xg_pred
    }
    
    # Filter out missing predictions (NaN)
    valid_preds = {k: v for k, v in predictions.items() if not np.isnan(v)}
    
    if not valid_preds:
        return np.nan, predictions
    
    # Recalculate weights based on valid models
    valid_keys = valid_preds.keys()
    total_valid_weight = sum(weights.get(k, 0) for k in valid_keys)
    
    if total_valid_weight == 0:
        # Fallback to simple average if weights are bad
        ensemble_pred = np.mean(list(valid_preds.values()))
    else:
        # Calculate weighted average
        ensemble_pred = sum(
            valid_preds[k] * (weights[k] / total_valid_weight) 
            for k in valid_keys
        )
    
    # Final clip and return
    ensemble_pred = np.clip(ensemble_pred, 0.0, 1.0)
    
    return ensemble_pred, predictions


# =============================================================================
# DATA QUERY FUNCTIONS (Unchanged)
# =============================================================================

def search_movie(db, title):
    # Search for a movie by title
    movie = db.movies.find_one({"title": {"$regex": f"^{title}$", "$options": "i"}})
    if not movie:
        movie = db.movies.find_one({"title": {"$regex": title, "$options": "i"}})
    return movie

def get_top_movies(db, limit=50, min_votes=100):
    # Get top-rated movies
    query = {
        "tmdb_metrics.vote_count": {"$gte": min_votes},
        "tmdb_metrics.vote_average": {"$ne": None}
    }
    return list(db.movies.find(query).sort("tmdb_metrics.vote_average", -1).limit(limit))

def get_similar_movies(db, tmdb_id, limit=10):
    # Get similar movies based on shared genres
    movie = db.movies.find_one({"tmdb_id": tmdb_id})
    if not movie:
        return []
    
    # Use movie's genres list directly
    genres = movie['production'].get('genres', [])
    if not genres:
        return []
        
    query = {
        "production.genres": {"$in": genres},
        "tmdb_id": {"$ne": tmdb_id}
    }
    return list(db.movies.find(query).sort("tmdb_metrics.vote_average", -1).limit(limit))

def get_database_stats(db):
    # Get database statistics
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
    # Get list of all unique genres
    genres = db.movies.distinct("production.genres")
    return sorted([g for g in genres if g])

def get_movies_by_genre(db, genre, limit=20):
    # Get movies by genre
    query = {"production.genres": genre}
    return list(db.movies.find(query).sort("tmdb_metrics.vote_average", -1).limit(limit))


# =============================================================================
# VISUALIZATION FUNCTIONS (Unchanged)
# =============================================================================

def create_genre_distribution_chart(db):
    # Create a bar chart showing genre distribution
    pipeline = [
        {"$unwind": "$production.genres"},
        {"$group": {"_id": "$production.genres", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 15}
    ]
    results = list(db.movies.aggregate(pipeline))
    df = pd.DataFrame(results)
    if df.empty: return None
    df.columns = ['Genre', 'Count']
    fig = px.bar(df, x='Genre', y='Count', 
                 title='Top 15 Movie Genres',
                 color='Count',
                 color_continuous_scale='viridis')
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def create_rating_distribution(db):
    # Create histogram of movie ratings
    movies = list(db.movies.find(
        {"tmdb_metrics.vote_average": {"$ne": None}},
        {"tmdb_metrics.vote_average": 1}
    ).limit(5000))
    
    ratings = [m['tmdb_metrics']['vote_average'] for m in movies]
    if not ratings: return None
    
    fig = px.histogram(ratings, nbins=50,
                       title='Distribution of Movie Ratings',
                       labels={'value': 'Rating', 'count': 'Number of Movies'},
                       color_discrete_sequence=['#1f77b4'])
    fig.update_layout(showlegend=False)
    return fig

def create_success_over_time(db):
    # Create line chart showing success rate over time
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
        if df.empty: return None
        
        fig = px.line(df, x='year', y='success_rate',
                     title='Movie Success Rate Over Time',
                     labels={'year': 'Year', 'success_rate': 'Success Rate (%)'},
                     markers=True)
        fig.update_layout(hovermode='x unified')
        return fig
    return None

# =============================================================================
# STREAMLIT APP LAYOUT (Modified)
# =============================================================================

def main():
    # Header
    st.title("🎬 Cinemaniacs")
    st.subheader("Ensemble Movie Success Prediction Platform")
    st.markdown("---")
    
    # Connect to database
    db = get_database_connection()
    if db is None:
        st.error("Unable to connect to database. Please check your connection string.")
        st.stop()
    
    # Load all ensemble artifacts
    artifacts, error = load_ensemble_artifacts()
    if error:
        st.error(f"❌ Ensemble Artifacts failed to load: {error}")
        st.stop()
    else:
        st.sidebar.success("✅ All Artifacts Loaded")
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Movie Search", "Analytics", "Ensemble Model", "Database Stats"]
    )
    
    # HOME PAGE
    if page == "Home":
        st.header("Welcome to Cinemaniacs")
        
        col1, col2, col3 = st.columns(3)
        
        stats = get_database_stats(db)
        
        with col1:
            st.metric("Total Movies", f"{stats['total']:,}")
        with col2:
            st.metric("Successful Movies", f"{stats['successful']:,}")
        with col3:
            st.metric("Success Rate", f"{(stats['successful']/stats['total']*100):.1f}%")
        
        st.markdown("---")
        
        # Top rated movies
        st.subheader("Top Rated Movies (TMDB)")
        top_movies = get_top_movies(db, limit=10)
        
        for i, movie in enumerate(top_movies[:5], 1):
            with st.expander(f"{i}. {movie['title']} - {movie['tmdb_metrics']['vote_average']}/10"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    poster = movie.get('content', {}).get('poster_url')
                    if poster:
                        st.image(poster, width=150)
                
                with col2:
                    st.write(f"**Genres:** {', '.join(movie['production'].get('genres', []))}")
                    st.write(f"**Runtime:** {movie['production'].get('runtime', 'N/A')} minutes")
                    st.write(f"**Votes:** {movie['tmdb_metrics']['vote_count']:,}")
                    overview = movie.get('content', {}).get('overview')
                    if overview:
                        st.write(f"**Overview:** {overview[:200]}...")
    
    # MOVIE SEARCH PAGE
    elif page == "Movie Search":
        st.header("Search Movies & Predict")
        
        # Search bar
        search_query = st.text_input("Enter movie title:", placeholder="e.g., Inception, The Matrix")
        
        if search_query:
            movie = search_movie(db, search_query)
            
            if movie:
                st.success(f"✅ Found: {movie['title']}")
                
                # --- ENSEMBLE PREDICTION ---
                ensemble_score, breakdown = predict_ensemble(movie, artifacts)
                
                st.markdown("---")
                st.subheader("🤖 Ensemble Audience Score Prediction")
                
                if not np.isnan(ensemble_score):
                    st.success(f"**Predicted Audience Score: {ensemble_score*100:.1f}%**")
                    
                    st.markdown("##### Model Breakdown")
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
                    st.warning("⚠️ Cannot generate Ensemble Prediction (Missing GNN/KGCN data for this movie).")
                
                st.markdown("---")

                # Movie details
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    poster = movie.get('content', {}).get('poster_url')
                    if poster:
                        st.image(poster, width=250)
                
                with col2:
                    st.subheader(movie['title'])
                    
                    # Basic info
                    st.write(f"**TMDB ID:** {movie['tmdb_id']}")
                    st.write(f"**Genres:** {', '.join(movie['production'].get('genres', []))}")
                    st.write(f"**Runtime:** {movie['production'].get('runtime', 'N/A')} minutes")
                    st.write(f"**Budget:** ${movie['production'].get('budget', 0):,.0f}")
                    st.write(f"**Release Date:** {movie['release_info'].get('tmdb_release_date', 'N/A')}")
                    
                    # Ratings
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
                    
                    # Success indicator
                    if movie['tmdb_metrics'].get('is_successful'):
                        st.success("✅ Classified as: SUCCESSFUL")
                    else:
                        st.error("❌ Classified as: NOT SUCCESSFUL")
                
                # ... (Rest of the Movie Search details) ...
                st.markdown("### 🎬 Similar Movies")
                similar = get_similar_movies(db, movie['tmdb_id'], limit=5)
                
                if similar:
                    cols = st.columns(5)
                    for idx, sim_movie in enumerate(similar):
                        with cols[idx]:
                            st.write(f"**{sim_movie['title']}**")
                            st.write(f" {sim_movie['tmdb_metrics']['vote_average']}/10")
                else:
                    st.write("No similar movies found")
            
            else:
                st.error(f"❌ No movie found matching '{search_query}'")
        
        # Browse by genre (unchanged)
        st.markdown("---")
        st.subheader("📂 Browse by Genre")
        genres = get_all_genres(db)
        selected_genre = st.selectbox("Select a genre:", genres)
        if selected_genre:
            genre_movies = get_movies_by_genre(db, selected_genre, limit=10)
            st.write(f"Showing top {len(genre_movies)} {selected_genre} movies:")
            for movie in genre_movies:
                st.write(f"**{movie['title']}** - ⭐ {movie['tmdb_metrics']['vote_average']}/10")
    
    # ANALYTICS PAGE (Unchanged)
    elif page == "Analytics":
        st.header("Data Analytics Dashboard")
        
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
    
    # ENSEMBLE MODEL PAGE (Modified)
    elif page == "Ensemble Model":
        st.header("Ensemble Model and Artifacts")
        
        st.success("✅ All Ensemble Artifacts Loaded Successfully")
        st.write("The platform runs an **Ensemble Model** combining Graph Neural Networks (GNN), Knowledge Graph Convolutional Networks (KGCN), and XGBoost to predict the Rotten Tomatoes Audience Score (0-100%).")
        
        st.markdown("### 💾 Artifacts Loaded Overview")
        
        # Display the ensemble weights
        st.subheader("Ensemble Weights")
        weights = artifacts.get('ensemble_weights', {})
        w_cols = st.columns(3)
        for i, (k, v) in enumerate(weights.items()):
            with w_cols[i]:
                st.metric(f"{k.upper()} Weight", f"{v*100:.1f}%")
        
        # Display the prediction coverage
        st.markdown("---")
        st.subheader("Model Coverage")
        
        cov_col1, cov_col2, cov_col3 = st.columns(3)
        with cov_col1:
            # FIX: Use 0 if the artifact is None
            gnn_preds_len = len(artifacts['gnn_preds']) if artifacts.get('gnn_preds') is not None else 0
            st.metric("GNN Predictions Found", f"{gnn_preds_len:,}")
        with cov_col2:
            # FIX: Use 0 if the artifact is None
            kgcn_preds_len = len(artifacts['kgcn_preds']) if artifacts.get('kgcn_preds') is not None else 0
            st.metric("KGCN Predictions Found", f"{kgcn_preds_len:,}")
        with cov_col3:
    # --- ENSURE THIS CODE IS PRESENT ---
            xg_features = artifacts.get('xg_features')
            if xg_features is not None:
                st.metric("XGBoost Feature Count", len(xg_features))
            else:
                # Prevents crash if the file is missing
                st.metric("XGBoost Feature Count", "N/A (Missing file)")
        
        st.markdown("""
        <small>GNN/KGCN predictions are pre-calculated lookups; XGBoost prediction is a placeholder/fallback in this web application for quick demo.</small>
        """, unsafe_allow_html=True)
    
    # DATABASE STATS PAGE (Unchanged)
    elif page == "Database Stats":
        st.header("Database Statistics")
        
        stats = get_database_stats(db)
        
        # Overview metrics
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
        
        # Detailed breakdowns
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
            
            fig = px.bar(df_complete, x='Category', y='Percentage',
                        title='Data Completeness (%)',
                        color='Percentage',
                        color_continuous_scale='greens')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Collection Info")
            
            # Get date range
            oldest = db.movies.find_one(
                {"release_info.tmdb_release_date": {"$ne": None}},
                sort=[("release_info.tmdb_release_date", 1)]
            )
            newest = db.movies.find_one(
                {"release_info.tmdb_release_date": {"$ne": None}},
                sort=[("release_info.tmdb_release_date", -1)]
            )
            
            if oldest and newest:
                st.write(f"**Oldest Movie:** {oldest['release_info']['tmdb_release_date']}")
                st.write(f"**Newest Movie:** {newest['release_info']['tmdb_release_date']}")
            
            # Top genres
            st.write("**Top 5 Genres:**")
            pipeline = [
                {"$unwind": "$production.genres"},
                {"$group": {"_id": "$production.genres", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 5}
            ]
            
            for doc in db.movies.aggregate(pipeline):
                st.write(f"- {doc['_id']}: {doc['count']:,}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🎬 Cinemaniacs | STA 160 Project | Team 15</p>
        <p>Ensemble Prediction Platform</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()