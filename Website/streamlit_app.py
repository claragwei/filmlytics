
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pymongo import MongoClient
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import os


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

# DATABASE CONNECTION

@st.cache_resource
def get_database_connection():
    """Connect to MongoDB and cache the connection"""
    try:
        client = MongoClient(MONGODB_URI)
        db = client.cinemaniacs
        # Test connection
        db.movies.count_documents({})
        return db
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        return None

# GNN MODEL DEFINITION

class GCN(torch.nn.Module):
    """Graph Convolutional Network for movie success prediction"""
    def __init__(self, in_feats, hidden, out_feats=1):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden)
        self.conv2 = GCNConv(hidden, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x.squeeze()

@st.cache_resource
def load_gnn_model():
    """Load the trained GNN model and graph data"""
    try:
        # Load the saved model weights
        model_path = './model_artifacts/gcn_weights_full_data_v4_scaled.pth'
        graph_path = './model_artifacts/full_movie_graph_v4_scaled.pt'
        
        if not os.path.exists(model_path) or not os.path.exists(graph_path):
            return None, None, "Model files not found"
        
        # Load graph data
        graph_data = torch.load(graph_path)
        
        # Initialize and load model
        model = GCN(in_feats=graph_data.num_features, hidden=64)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        return model, graph_data, None
    except Exception as e:
        return None, None, str(e)

# DATA QUERY FUNCTIONS

def search_movie(db, title):
    """Search for a movie by title"""
    # Try exact match first
    movie = db.movies.find_one({"title": {"$regex": f"^{title}$", "$options": "i"}})
    
    # Try partial match if exact not found
    if not movie:
        movie = db.movies.find_one({"title": {"$regex": title, "$options": "i"}})
    
    return movie

def get_top_movies(db, limit=50, min_votes=100):
    """Get top-rated movies"""
    query = {
        "tmdb_metrics.vote_count": {"$gte": min_votes},
        "tmdb_metrics.vote_average": {"$ne": None}
    }
    return list(db.movies.find(query).sort("tmdb_metrics.vote_average", -1).limit(limit))

def get_movies_by_genre(db, genre, limit=20):
    """Get movies by genre"""
    query = {"production.genres": genre}
    return list(db.movies.find(query).sort("tmdb_metrics.vote_average", -1).limit(limit))

def get_similar_movies(db, tmdb_id, limit=10):
    """Get similar movies based on shared genres"""
    movie = db.movies.find_one({"tmdb_id": tmdb_id})
    if not movie:
        return []
    
    query = {
        "production.genres": {"$in": movie['production']['genres']},
        "tmdb_id": {"$ne": tmdb_id}
    }
    return list(db.movies.find(query).sort("tmdb_metrics.vote_average", -1).limit(limit))

def get_database_stats(db):
    """Get database statistics"""
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
    """Get list of all unique genres"""
    genres = db.movies.distinct("production.genres")
    return sorted([g for g in genres if g])

# VISUALIZATION FUNCTIONS

def create_genre_distribution_chart(db):
    """Create a bar chart showing genre distribution"""
    pipeline = [
        {"$unwind": "$production.genres"},
        {"$group": {"_id": "$production.genres", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 15}
    ]
    
    results = list(db.movies.aggregate(pipeline))
    df = pd.DataFrame(results)
    df.columns = ['Genre', 'Count']
    
    fig = px.bar(df, x='Genre', y='Count', 
                 title='Top 15 Movie Genres',
                 color='Count',
                 color_continuous_scale='viridis')
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def create_rating_distribution(db):
    """Create histogram of movie ratings"""
    movies = list(db.movies.find(
        {"tmdb_metrics.vote_average": {"$ne": None}},
        {"tmdb_metrics.vote_average": 1}
    ).limit(5000))
    
    ratings = [m['tmdb_metrics']['vote_average'] for m in movies]
    
    fig = px.histogram(ratings, nbins=50,
                       title='Distribution of Movie Ratings',
                       labels={'value': 'Rating', 'count': 'Number of Movies'},
                       color_discrete_sequence=['#1f77b4'])
    fig.update_layout(showlegend=False)
    return fig

def create_success_over_time(db):
    """Create line chart showing success rate over time"""
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
        df = df[df['year'].astype(int) >= 2000]  # Filter for recent years
        
        fig = px.line(df, x='year', y='success_rate',
                     title='Movie Success Rate Over Time',
                     labels={'year': 'Year', 'success_rate': 'Success Rate (%)'},
                     markers=True)
        fig.update_layout(hovermode='x unified')
        return fig
    return None

# STREAMLIT APP LAYOUT

def main():
    # Header
    st.title("🎬 Cinemaniacs")
    st.subheader("Movie Success Prediction Platform")
    st.markdown("---")
    
    # Connect to database
    db = get_database_connection()
    if db is None:
        st.error("Unable to connect to database. Please check your connection string.")
        st.stop()
    
    # Load GNN model
    model, graph_data, error = load_gnn_model()
    if error:
        st.warning(f"⚠️ GNN Model not loaded: {error}. Some features may be unavailable.")
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Movie Search", "Analytics", "GNN Model", "Database Stats"]
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
        st.subheader("Top Rated Movies")
        top_movies = get_top_movies(db, limit=10)
        
        for i, movie in enumerate(top_movies[:5], 1):
            with st.expander(f"{i}. {movie['title']} - {movie['tmdb_metrics']['vote_average']}/10"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if movie['content'].get('poster_url'):
                        st.image(movie['content']['poster_url'], width=150)
                
                with col2:
                    st.write(f"**Genres:** {', '.join(movie['production']['genres'])}")
                    st.write(f"**Runtime:** {movie['production'].get('runtime', 'N/A')} minutes")
                    st.write(f"**Votes:** {movie['tmdb_metrics']['vote_count']:,}")
                    if movie['content'].get('overview'):
                        st.write(f"**Overview:** {movie['content']['overview'][:200]}...")
    
    # MOVIE SEARCH PAGE
    elif page == "Movie Search":
        st.header("Search Movies")
        
        # Search bar
        search_query = st.text_input("Enter movie title:", placeholder="e.g., Inception, The Matrix")
        
        if search_query:
            movie = search_movie(db, search_query)
            
            if movie:
                st.success(f"✅ Found: {movie['title']}")
                
                # Movie details
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if movie['content'].get('poster_url'):
                        st.image(movie['content']['poster_url'], width=250)
                
                with col2:
                    st.subheader(movie['title'])
                    
                    # Basic info
                    st.write(f"**TMDB ID:** {movie['tmdb_id']}")
                    st.write(f"**Genres:** {', '.join(movie['production']['genres'])}")
                    st.write(f"**Runtime:** {movie['production'].get('runtime', 'N/A')} minutes")
                    st.write(f"**Budget:** ${movie['production'].get('budget', 0):,.0f}")
                    st.write(f"**Release Date:** {movie['release_info'].get('tmdb_release_date', 'N/A')}")
                    
                    # Ratings
                    st.markdown("### Ratings")
                    rating_col1, rating_col2, rating_col3 = st.columns(3)
                    
                    with rating_col1:
                        st.metric("TMDB Rating", f"{movie['tmdb_metrics']['vote_average']}/10")
                    
                    with rating_col2:
                        if movie['rotten_tomatoes'].get('critic_score'):
                            st.metric("RT Critics", movie['rotten_tomatoes']['critic_score'])
                        else:
                            st.metric("RT Critics", "N/A")
                    
                    with rating_col3:
                        if movie['rotten_tomatoes'].get('audience_score'):
                            st.metric("RT Audience", movie['rotten_tomatoes']['audience_score'])
                        else:
                            st.metric("RT Audience", "N/A")
                    
                    # Success indicator
                    if movie['tmdb_metrics'].get('is_successful'):
                        st.success("✅ Classified as: SUCCESSFUL")
                    else:
                        st.error("❌ Classified as: NOT SUCCESSFUL")
                
                # Overview
                st.markdown("### Overview")
                st.write(movie['content'].get('overview', 'No overview available'))
                
                # Cast and Crew
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🎭 Cast")
                    cast = movie['people'].get('cast', [])
                    if cast:
                        st.write(", ".join(cast[:10]))
                    else:
                        st.write("No cast information available")
                
                with col2:
                    st.markdown("### 🎬 Directors")
                    directors = movie['people'].get('directors', [])
                    if directors:
                        st.write(", ".join(directors))
                    else:
                        st.write("No director information available")
                
                # Trailer info
                if movie['trailer'].get('trailer_url_youtube'):
                    st.markdown("### 🎥 Trailer Metrics")
                    tcol1, tcol2, tcol3, tcol4 = st.columns(4)
                    
                    with tcol1:
                        views = movie['trailer']['metrics'].get('view_count', 0)
                        st.metric("Views", f"{views:,}")
                    
                    with tcol2:
                        likes = movie['trailer']['metrics'].get('like_count', 0)
                        st.metric("Likes", f"{likes:,}")
                    
                    with tcol3:
                        comments = movie['trailer']['metrics'].get('comment_count', 0)
                        st.metric("Comments", f"{comments:,}")
                    
                    with tcol4:
                        days = movie['release_info'].get('days_until_release', 0)
                        st.metric("Days Until Release", days)
                
                # Similar movies
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
        
        # Browse by genre
        st.markdown("---")
        st.subheader("📂 Browse by Genre")
        
        genres = get_all_genres(db)
        selected_genre = st.selectbox("Select a genre:", genres)
        
        if selected_genre:
            genre_movies = get_movies_by_genre(db, selected_genre, limit=10)
            
            st.write(f"Showing top {len(genre_movies)} {selected_genre} movies:")
            
            for movie in genre_movies:
                st.write(f"**{movie['title']}** - ⭐ {movie['tmdb_metrics']['vote_average']}/10")
    
    # ANALYTICS PAGE
    elif page == "Analytics":
        st.header("Data Analytics Dashboard")
        
        tab1, tab2, tab3 = st.tabs(["Genre Analysis", "Rating Distribution", "Success Trends"])
        
        with tab1:
            st.subheader("Genre Distribution")
            fig = create_genre_distribution_chart(db)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Rating Distribution")
            fig = create_rating_distribution(db)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Success Rate Over Time")
            fig = create_success_over_time(db)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient data for time series analysis")
    
    # GNN MODEL PAGE
    elif page == "GNN Model":
        st.header("Graph Neural Network Model")
        
        if model is None or graph_data is None:
            st.error("❌ GNN model not available. Please ensure model files are in './model_artifacts/' directory.")
            st.info("Expected files:\n- gcn_weights_full_data_v4_scaled.pth\n- full_movie_graph_v4_scaled.pt")
        else:
            st.success("✅ GNN Model Loaded Successfully")
            
            # Model info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Graph Nodes", f"{graph_data.num_nodes:,}")
            with col2:
                st.metric("Graph Edges", f"{graph_data.num_edges:,}")
            with col3:
                st.metric("Node Features", graph_data.num_features)
            
            st.markdown("---")
            
            # Model description
            st.subheader("📖 About the Model")
            st.write("""
            Our Graph Convolutional Network (GNN) predicts movie success by learning from relationships between movies.
            
            **Features used:**
            - Production metadata (budget, runtime, vote counts)
            - Sentiment scores (reviews and descriptions)
            - Trailer metrics (views, likes, comments, timing)
            - Genre information (multi-hot encoded)
            
            **Graph structure:**
            - Nodes: Movies
            - Edges: Movies connected by shared genres
            - Target: Rotten Tomatoes Critic Score
            """)
            
            # Model architecture
            with st.expander("🏗️ Model Architecture"):
                st.code("""
class GCN(torch.nn.Module):
    def __init__(self, in_feats, hidden, out_feats=1):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden)  # 64 hidden units
        self.conv2 = GCNConv(hidden, out_feats)  # Output layer
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x.squeeze()
                """, language="python")
            
            # Feature importance
            st.markdown("### 🎯 Key Features")
            
            feature_cols = [
                'Budget', 'Runtime', 'Vote Count', 'Vote Average',
                'Review Sentiment', 'Description Sentiment',
                'View Count', 'Like Count', 'Comment Count',
                'Days Until Release', 'Before Release Flag'
            ]
            
            st.write("The model uses the following features:")
            for feat in feature_cols:
                st.write(f"- {feat}")
    
    # DATABASE STATS PAGE
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
        <p>Predicting Movie Success with Graph Neural Networks</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import certifi
import os
import joblib # For loading XGBoost model and scaler
import json   # For loading feature list
import pickle # For loading stacking model

# CONFIGURATION

# MongoDB Connection
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
        st.error(f"❌ Database connection failed: {e}")
        return None

# =============================================================================
# ARTIFACT LOADING AND ENSEMBLE PREDICTION SETUP
# =============================================================================

@st.cache_resource
def load_ensemble_artifacts():
    """Load all models, scalers, and prediction dataframes including stacking model."""

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
        # XGBoost Predictions (Lookup)
        'xgb_preds': 'xgb_preds_all_movies.csv',
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
                # Handle different column names for predictions
                if 'pred_audience_score' in df.columns:
                    artifacts[key] = df.set_index('tmdb_id')['pred_audience_score']
                elif 'predicted_audience_score' in df.columns:
                    artifacts[key] = df.set_index('tmdb_id')['predicted_audience_score']
                else:
                    # Fallback: use first numeric column after tmdb_id
                    artifacts[key] = df.set_index('tmdb_id').iloc[:, 0]

        except Exception as e:
            return None, f"Error loading {filename}: {e}"

    # --- Load Stacking Meta-Model ---
    stacking_path = os.path.join(artifact_dir, 'stacking_meta_model.pkl')
    if os.path.exists(stacking_path):
        try:
            with open(stacking_path, 'rb') as f:
                artifacts['stacking_model'] = pickle.load(f)
        except Exception as e:
            st.warning(f"⚠️ Could not load stacking model: {e}")
            artifacts['stacking_model'] = None
    else:
        artifacts['stacking_model'] = None

    # --- Load Ensemble Metadata ---
    meta_path = os.path.join(artifact_dir, 'ensemble_weights.json')
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                artifacts['ensemble_meta'] = json.load(f)
        except:
            artifacts['ensemble_meta'] = {}
    else:
        artifacts['ensemble_meta'] = {}

    # --- Fallback Ensemble weights (for weighted average if stacking fails) ---
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

    # 1. Get all predictions from lookup tables
    gnn_pred = safe_get_prediction(artifacts.get('gnn_preds'), tmdb_id)
    kgcn_pred = safe_get_prediction(artifacts.get('kgcn_preds'), tmdb_id)
    xgb_pred = safe_get_prediction(artifacts.get('xgb_preds'), tmdb_id)

    # Store predictions for display
    predictions = {
        'gnn': gnn_pred,
        'kgcn': kgcn_pred,
        'xg': xgb_pred
    }

    # 2. Check if we have the stacking model and all 3 predictions
    stacking_model = artifacts.get('stacking_model')
    has_all_preds = not (np.isnan(gnn_pred) or np.isnan(kgcn_pred) or np.isnan(xgb_pred))

    if stacking_model is not None and has_all_preds:
        # USE STACKING MODEL (best accuracy)
        X = np.array([[gnn_pred, kgcn_pred, xgb_pred]])
        ensemble_pred = float(np.clip(stacking_model.predict(X)[0], 0, 1))
        return ensemble_pred, predictions

    # 3. FALLBACK: Weighted average (if stacking not available)
    weights = artifacts.get('ensemble_weights', {'gnn': 0.33, 'kgcn': 0.34, 'xg': 0.33})

    # Filter out missing predictions
    valid_preds = {k: v for k, v in predictions.items() if not np.isnan(v)}

    if not valid_preds:
        # Last resort: use TMDB vote average
        tmdb_avg = movie_data.get('tmdb_metrics', {}).get('vote_average')
        if tmdb_avg is not None:
            return float(tmdb_avg) / 10.0, predictions
        return np.nan, predictions

    # Calculate weighted average with available predictions
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
    """
    Get a list of UNIQUE movie titles, sorted by vote count (popularity), 
    for the dropdown using aggregation for deduplication.
    """
    try:
        pipeline = [
            # 1. Match: Filter for relevant documents
            {"$match": {
                "title": {"$ne": None}, 
                "tmdb_id": {"$ne": None},
                "tmdb_metrics.vote_count": {"$gte": 1}
            }},
            
            # 2. Deduplicate: Group by tmdb_id, keeping the maximum vote count
            #    and the corresponding title (since titles should be consistent 
            #    per tmdb_id).
            {"$group": {
                "_id": "$tmdb_id",
                "title": {"$first": "$title"},  # Keep the title
                "vote_count": {"$max": "$tmdb_metrics.vote_count"} # Get the highest vote count among duplicates
            }},
            
            # 3. Sort: Sort the unique results by vote count (highest first)
            {"$sort": {"vote_count": -1}},
            
            # 4. Project and Extract Titles
            {"$project": {"_id": 0, "title": 1}}
        ]
        
        # Execute the aggregation
        unique_titles = list(_db.movies.aggregate(pipeline))
        
        # Extract and return the titles
        return [doc['title'] for doc in unique_titles]
        
    except Exception as e:
        st.error(f"Error fetching unique movie titles: {e}")
        return []

def search_movie(db, title):
    # Search for a movie by title
    movie = db.movies.find_one({"title": {"$regex": f"^{title}$", "$options": "i"}})
    if not movie:
        movie = db.movies.find_one({"title": {"$regex": title, "$options": "i"}})
    return movie

def get_top_movies(db, limit=50, min_votes=1000):
    # Get top-rated movies
    query = {
        "tmdb_metrics.vote_count": {"$gte": min_votes},
        "tmdb_metrics.vote_average": {"$ne": None}
    }
    return list(db.movies.find(query).sort("tmdb_metrics.vote_average", -1).limit(limit))

def get_similar_movies(db, tmdb_id, limit=10, min_votes=1000):
    """
    Get unique similar movies based on shared genres (exact match), 
    filtered by vote count, and sorted by average.
    """
    # 1. Find the reference movie
    movie = db.movies.find_one({"tmdb_id": tmdb_id})
    if not movie:
        return []
    
    genres = movie['production'].get('genres', [])
    if not genres:
        return []
        
    genre_count = len(genres)
        
    # 2. Build the Aggregation Pipeline
    pipeline = [
        # A. Filter: Match the exact genre set, minimum votes, and ensure it's not the original movie
        {"$match": {
            "production.genres": {
                "$all": genres,
                "$size": genre_count
            },
            "tmdb_id": {"$ne": tmdb_id},
            "tmdb_metrics.vote_count": {"$gte": min_votes}, 
            "tmdb_metrics.vote_average": {"$ne": None} 
        }},
        
        # B. Deduplicate: Group by the unique identifier (tmdb_id)
        #    We keep the entire document of the first instance found using "$$ROOT".
        {"$group": {
            "_id": "$tmdb_id",
            "unique_movie": {"$first": "$$ROOT"}
        }},
        
        # C. Project: Restore the document structure
        {"$replaceRoot": {"newRoot": "$unique_movie"}},
        
        # D. Sort: Sort the unique results by vote average (highest first)
        {"$sort": {"tmdb_metrics.vote_average": -1}},
        
        # E. Limit: Apply the final limit
        {"$limit": limit}
    ]
    
    # Execute the aggregation pipeline and return the list of documents
    return list(db.movies.aggregate(pipeline))

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

def get_movies_by_genre(db, genre, limit=20, min_votes=1000):
    pipeline = [
        # 1. Filter: Match the genre and minimum vote count/rating
        {"$match": {
            "production.genres": genre,
            "tmdb_metrics.vote_count": {"$gte": min_votes}, 
            "tmdb_metrics.vote_average": {"$ne": None}
        }},
        
        # 2. Deduplicate: Group by the unique identifier (tmdb_id). 
        #    We use '$first' to keep the data from the first document found 
        #    for this movie, including the entire original document.
        {"$group": {
            "_id": "$tmdb_id",
            "unique_movie": {"$first": "$$ROOT"} # $$ROOT keeps the whole document
        }},
        
        # 3. Project: Restore the document structure to what 'find' returns 
        #    and access the fields needed for sorting.
        {"$replaceRoot": {"newRoot": "$unique_movie"}},
        
        # 4. Sort: Sort the unique results by vote average (highest first)
        {"$sort": {"tmdb_metrics.vote_average": -1}},
        
        # 5. Limit: Apply the final limit
        {"$limit": limit}
    ]
    
    # Execute the aggregation pipeline and return the list of documents
    return list(db.movies.aggregate(pipeline))


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
        
        # 1. Fetch all titles for the dropdown
        all_titles = get_all_movie_titles(db)

        # 2. Dropdown search menu (using st.selectbox for simplicity, which works as an autocomplete)
        selected_title = st.selectbox(
            "Select or Type a Movie Title:", 
            options=["-- Select a Movie --"] + all_titles,
            index=0
        )
        
        # Check if the user selected a movie (and not the placeholder)
        search_query = selected_title if selected_title != "-- Select a Movie --" else None
        
        if search_query:
            # The search logic remains the same, but now uses the selected title
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
    
    # ENSEMBLE MODEL PAGE (Updated for Stacking)
    elif page == "Ensemble Model":
        st.header("🧠 Stacking Ensemble Model")

        # Check if stacking model is loaded
        stacking_model = artifacts.get('stacking_model') if artifacts else None
        ensemble_meta = artifacts.get('ensemble_meta', {}) if artifacts else {}

        if stacking_model is not None:
            st.success("✅ Stacking Meta-Learner Loaded Successfully")
        else:
            st.warning("⚠️ Stacking model not found. Using weighted average fallback.")

        st.write("""
        The platform uses a **Stacking Ensemble** that combines predictions from three base models
        using a learned meta-learner to predict Rotten Tomatoes Audience Score (0-100%).

        **Base Models:**
        - **GNN** (Graph Neural Network): Learns from movie similarity graph
        - **KGCN** (Knowledge Graph Convolutional Network): Uses genre/director/cast relationships
        - **XGBoost**: Gradient boosting on 156 engineered features
        """)

        st.markdown("---")

        # Stacking Model Info
        if stacking_model is not None:
            st.subheader("🎯 Stacking Meta-Learner Performance")

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
            **How Stacking Works:**
            1. Each base model (GNN, KGCN, XGBoost) makes a prediction
            2. The Gradient Boosting meta-learner takes all 3 predictions as input
            3. It learns the optimal non-linear combination
            4. Returns the final ensemble prediction
            """)

        # Individual Model Performance
        st.markdown("---")
        st.subheader("📊 Individual Model Performance")

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

        # Model Coverage
        st.markdown("---")
        st.subheader("🗂️ Prediction Coverage")

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

        # Comparison Table
        st.markdown("---")
        st.subheader("📈 Original vs Stacking Comparison")

        comparison_data = {
            'Method': ['Original (Weighted Avg)', 'Stacking (Gradient Boosting)'],
            'RMSE': [0.1452, 0.0996],
            'MAE': [0.1170, 0.0669],
            'Within ±10%': ['48.9%', '80.2%'],
            'Within ±5%': ['24.2%', '54.5%']
        }
        st.dataframe(pd.DataFrame(comparison_data), hide_index=True, use_container_width=True)

        st.caption("Stacking achieves 31.4% better RMSE and predicts 80% of movies within 10% of actual score.")
    
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
        
        pipeline = [
            # 1. Match: Filter out empty, null, or obviously non-date values
            {"$match": {
                "release_info.tmdb_release_date": {
                    "$ne": None, 
                    # Regex: Ensures the value starts with four digits (YYYY)
                    "$regex": "^[0-9]{4}",
                    # Filter out short strings that are clearly not dates (e.g., "TLA")
                    "$type": "string" 
                }
            }},
            # 2. Group: Find the minimum and maximum *valid* release date strings
            {"$group": {
                "_id": None,
                "oldest_date": {"$min": "$release_info.tmdb_release_date"},
                "newest_date": {"$max": "$release_info.tmdb_release_date"}
            }}
        ]
        
        results = list(db.movies.aggregate(pipeline))
        
        if results:
            dates = results[0]
            
            # Find the full movie documents corresponding to the min/max dates
            oldest_movie = db.movies.find_one({"release_info.tmdb_release_date": dates["oldest_date"]})
            newest_movie = db.movies.find_one({"release_info.tmdb_release_date": dates["newest_date"]})
            
            if oldest_movie and newest_movie:
                st.write(f"**Oldest Movie:** {oldest_movie['title']} ({dates['oldest_date']})")
                st.write(f"**Newest Movie:** {newest_movie['title']} ({dates['newest_date']})")
            else:
                st.info("Could not retrieve movies for the calculated date range.")
        else:
            st.info("No valid release dates found in the database.")
            
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