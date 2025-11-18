
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
    main()