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
import joblib
import json
import pickle

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Predicting Movie Audience Scores Using Graph-Based Modeling",
    page_icon="🎬",
    layout="wide"
)

# Navigation key FIX — ensures no Streamlit crashes
if "nav" not in st.session_state:
    st.session_state["nav"] = "Home"

# -----------------------------------------------------------------------------
# GLOBAL CSS (clean + modern)
# -----------------------------------------------------------------------------

st.markdown("""
<style>
.main-title {
    font-size: 2.4rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.sub-title {
    font-size: 1.15rem;
    color: #666;
    margin-bottom: 1.3rem;
}
.hero-btn button {
    border-radius: 12px;
    padding: 0.6rem 1rem;
    font-size: 1rem;
}
.section-header {
    font-size: 1.4rem;
    font-weight: 600;
    margin-top: 0.8rem;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# DATABASE CONNECTION
# -----------------------------------------------------------------------------

@st.cache_resource
def get_db():
    uri = st.secrets.get(
        "MONGODB_URI",
        "mongodb+srv://cinemaniacs:filmlytics@filmlytics.1emhcue.mongodb.net/?appName=filmlytics"
    )
    try:
        client = MongoClient(uri, server_api=ServerApi("1"), tlsCAFile=certifi.where())
        db = client["cinemaniacs"]
        db.movies.count_documents({})
        return db
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        return None


# -----------------------------------------------------------------------------
# LOAD ARTIFACTS
# -----------------------------------------------------------------------------

@st.cache_resource
def load_artifacts():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    artifact_dir = os.path.join(script_dir, "model_artifacts")

    files = {
        "gnn_preds": "gnn_preds_all_movies.csv",
        "kgcn_preds": "kgcn_preds_all_movies.csv",
        "xgb_preds": "xgb_preds_all_movies.csv",
        "scaler": "movie_feature_scaler_diversity.pkl",
        "stacking_model": "stacking_meta_model.pkl",
        "ensemble_meta": "ensemble_weights.json"
    }

    artifacts = {}

    for key, filename in files.items():
        path = os.path.join(artifact_dir, filename)
        if not os.path.exists(path):
            st.warning(f"⚠ Missing file: {filename}")
            artifacts[key] = None
            continue

        try:
            if filename.endswith(".csv"):
                df = pd.read_csv(path)
                df["tmdb_id"] = df["tmdb_id"].astype(int)
                col = (
                    "pred_audience_score" if "pred_audience_score" in df.columns else
                    "predicted_audience_score" if "predicted_audience_score" in df.columns else
                    "ensemble_pred"
                )
                artifacts[key] = df.set_index("tmdb_id")[col]

            elif filename.endswith(".pkl"):
                with open(path, "rb") as f:
                    artifacts[key] = pickle.load(f)

            elif filename.endswith(".json"):
                with open(path, "r") as f:
                    artifacts[key] = json.load(f)

        except Exception as e:
            st.error(f"Error loading {filename}: {e}")

    # Default fallback weights
    artifacts["ensemble_weights"] = {"gnn": 0.33, "kgcn": 0.34, "xg": 0.33}

    return artifacts


def safe_get(preds, tmdb_id):
    if preds is None:
        return np.nan
    val = preds.get(tmdb_id, np.nan)
    if isinstance(val, pd.Series):  # multiple rows
        val = val.iloc[0]
    return float(val) if pd.notna(val) else np.nan


def predict(movie, artifacts):
    tmdb = movie.get("tmdb_id")
    gnn = safe_get(artifacts["gnn_preds"], tmdb)
    kgcn = safe_get(artifacts["kgcn_preds"], tmdb)
    xgb = safe_get(artifacts["xgb_preds"], tmdb)

    base = {"gnn": gnn, "kgcn": kgcn, "xg": xgb}

    # STACKING MODEL
    if artifacts["stacking_model"] is not None and all(pd.notna(list(base.values()))):
        X = np.array([[gnnn, kgcn, xgb]])
        pred = float(artifacts["stacking_model"].predict(X)[0])
        return np.clip(pred, 0, 1), base

    # FALLBACK WEIGHTED AVERAGE
    valid = {k: v for k, v in base.items() if pd.notna(v)}
    if not valid:
        return np.nan, base

    w = artifacts["ensemble_weights"]
    Z = sum(w[k] for k in valid)
    pred = sum(valid[k] * (w[k] / Z) for k in valid)
    return np.clip(pred, 0, 1), base


# -----------------------------------------------------------------------------
# QUERY HELPERS
# -----------------------------------------------------------------------------

@st.cache_data
def all_titles(db):
    pipeline = [
        {"$match": {"tmdb_id": {"$ne": None}, "tmdb_metrics.vote_count": {"$gte": 1}}},
        {"$group": {"_id": "$tmdb_id", "title": {"$first": "$title"}}},
        {"$sort": {"title": 1}},
    ]
    return [x["title"] for x in db.movies.aggregate(pipeline)]


def search_movie(db, title):
    return db.movies.find_one({"title": {"$regex": f"^{title}$", "$options": "i"}})


def get_genre_neighbors(db, genre, limit=15):
    pipeline = [
        {"$match": {"production.genres": genre}},
        {"$limit": limit}
    ]
    return list(db.movies.aggregate(pipeline))


# -----------------------------------------------------------------------------
# GRAPH VISUALIZATION
# -----------------------------------------------------------------------------

def build_graph(db, movie, max_neighbors=15):
    genres = movie.get("production", {}).get("genres", [])
    if not genres:
        return None

    genre = genres[0]
    neighbors = get_genre_neighbors(db, genre, limit=max_neighbors)

    G = nx.Graph()
    G.add_node(movie["tmdb_id"], title=movie["title"], center=True)

    for m in neighbors:
        mid = m["tmdb_id"]
        if mid != movie["tmdb_id"]:
            G.add_node(mid, title=m["title"], center=False)
            G.add_edge(movie["tmdb_id"], mid)

    pos = nx.spring_layout(G, seed=42)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_labels = [G.nodes[n]["title"] for n in G.nodes()]
    node_colors = ["#fbbf24" if G.nodes[n]["center"] else "#60a5fa" for n in G.nodes()]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(width=1, color="#aaa"), hoverinfo="none"))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=node_labels, textposition="top center",
        marker=dict(size=[18 if c == "#fbbf24" else 10 for c in node_colors],
                    color=node_colors),
        hoverinfo="text"
    ))
    fig.update_layout(height=500, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


# -----------------------------------------------------------------------------
# HOME PAGE
# -----------------------------------------------------------------------------

def page_home(db):
    st.markdown("<div class='main-title'>Predicting Movie Audience Scores Using Graph-Based Modeling</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Filmytics — Combining graph learning, metadata, and ensemble modeling for film audience prediction.</div>", unsafe_allow_html=True)

    # Nav buttons under title
    cols = st.columns(4)
    buttons = ["Home", "Movie Search", "Compare Movies", "Analytics Dashboard"]
    for label, col in zip(buttons, cols):
        if col.button(label, key=f"nav_{label}"):
            st.session_state["nav"] = label
            st.rerun()

    cols2 = st.columns(3)
    for label, col in zip(["Modeling", "Visual Graph Explorer", "Acknowledgements"], cols2):
        if col.button(label, key=f"nav_{label}"):
            st.session_state["nav"] = label
            st.rerun()

    st.markdown("---")

    # -----------------------------
    # CLEAN — Your Updated Intro
    # -----------------------------

    st.header("Filmytics — Predicting Audience Scores")
    st.write("""
Filmytics is a data-driven system that predicts audience scores for movies **before release**.
Studios care because reception affects marketing, streaming placement, and long-term financial performance.
For fans and researchers, it reveals which kinds of films resonate — and whether representation matters.

We gather large-scale public data, learn patterns from 66k+ films, and estimate audience reactions for upcoming releases.
""")

    st.header("Our Approach")
    st.write("""
1. **Collect data from TMDB, Rotten Tomatoes, YouTube** (metadata, critic text, trailer engagement).  
2. **Clean + merge into MongoDB**, compute new metrics (sentiment, female cast %, etc.).  
3. **Engineer 150+ features** including metadata, engagement, diversity signals.  
4. **Train three models:**  
   - GNN (movie similarity graph)  
   - KGCN (entity knowledge graph)  
   - XGBoost (feature-based)  
5. **Combine outputs using a stacking meta-learner**, improving accuracy beyond any individual model.
""")

    st.info("Workflow: TMDB + RT + YouTube → Clean/Merge → Feature Engineering → {GNN, KGCN, XGBoost} → Stacking Model → Prediction")

    st.header("Main Results")
    st.write("""
- XGBoost RMSE: **0.110**  
- KGCN RMSE: **0.171**  
- GNN RMSE: **0.195**  
- Stacking Ensemble RMSE: **0.1085** → 31% better than original weighting  
- **80.2%** of predictions within ±10% of true score  
""")


# -----------------------------------------------------------------------------
# MOVIE SEARCH PAGE
# -----------------------------------------------------------------------------

def page_movie_search(db, artifacts):
    st.title("Movie Search & Predictions")

    titles = all_titles(db)
    choice = st.selectbox("Search for a movie:", ["-- Select --"] + titles)

    if choice == "-- Select --":
        return

    movie = search_movie(db, choice)
    if movie is None:
        st.error("Movie not found.")
        return

    st.success(f"Found: {movie['title']}")

    # Predictions
    pred, base = predict(movie, artifacts)
    st.subheader("Predicted Audience Score")
    if pd.notna(pred):
        st.metric("Ensemble Score", f"{pred*100:.1f}%")
    else:
        st.warning("No prediction available.")

    # Movie info
    st.markdown("---")
    st.subheader(movie["title"])

    cols = st.columns([1,2])
    with cols[0]:
        poster = movie.get("content", {}).get("poster_url")
        if poster:
            st.image(poster, width=250)

    with cols[1]:
        st.write(f"**Genres:** {', '.join(movie['production'].get('genres', []))}")
        st.write(f"**Runtime:** {movie['production'].get('runtime','N/A')} minutes")
        st.write(f"**Budget:** ${movie['production'].get('budget',0):,}")
        st.write(f"**TMDB Score:** {movie['tmdb_metrics'].get('vote_average')}/10")

    st.markdown("---")
    st.subheader("Local Graph Neighborhood")

    fig = build_graph(db, movie)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Graph unavailable for this movie.")


# -----------------------------------------------------------------------------
# COMPARE MOVIES
# -----------------------------------------------------------------------------

def page_compare(db, artifacts):
    st.title("Compare Movies")

    titles = all_titles(db)
    c1, c2 = st.columns(2)
    t1 = c1.selectbox("Movie 1:", ["-- Select --"] + titles)
    t2 = c2.selectbox("Movie 2:", ["-- Select --"] + titles)

    if "--" in (t1, t2):
        return

    m1 = search_movie(db, t1)
    m2 = search_movie(db, t2)

    pred1, _ = predict(m1, artifacts)
    pred2, _ = predict(m2, artifacts)

    st.subheader("Prediction Comparison")
    c1, c2 = st.columns(2)
    c1.metric(t1, f"{pred1*100:.1f}%")
    c2.metric(t2, f"{pred2*100:.1f}%")


# -----------------------------------------------------------------------------
# ANALYTICS DASHBOARD
# -----------------------------------------------------------------------------

def page_analytics(db):
    st.title("Analytics Dashboard")

    tab1, tab2, tab3 = st.tabs(["Genres", "Ratings", "Success Trends"])

    with tab1:
        st.subheader("Genre Distribution")
        pipeline = [
            {"$unwind": "$production.genres"},
            {"$group": {"_id": "$production.genres", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        data = list(db.movies.aggregate(pipeline))
        df = pd.DataFrame(data).rename(columns={"_id": "Genre"})
        st.plotly_chart(px.bar(df, x="Genre", y="count"), use_container_width=True)

    with tab2:
        st.subheader("Rating Distribution")
        movies = db.movies.find({}, {"tmdb_metrics.vote_average": 1}).limit(5000)
        ratings = [m["tmdb_metrics"]["vote_average"] for m in movies if m["tmdb_metrics"].get("vote_average")]
        st.plotly_chart(px.histogram(ratings, nbins=40), use_container_width=True)

    with tab3:
        st.subheader("Success Over Time")
        pipeline = [
            {"$match": {"release_info.tmdb_release_date": {"$regex": "^[0-9]{4}"}}},
            {"$project": {"year": {"$substr": ["$release_info.tmdb_release_date", 0, 4]},
                          "is_successful": "$tmdb_metrics.is_successful"}},
            {"$group": {"_id": "$year", "total": {"$sum": 1},
                        "successful": {"$sum": {"$cond": ["$is_successful", 1, 0]}}}}
        ]
        df = pd.DataFrame(db.movies.aggregate(pipeline))
        df["success_rate"] = df["successful"] / df["total"] * 100
        st.plotly_chart(px.line(df, x="_id", y="success_rate"), use_container_width=True)


# -----------------------------------------------------------------------------
# MODELING PAGE
# -----------------------------------------------------------------------------

def page_modeling(artifacts):
    st.title("Modeling Overview")

    st.header("Base Models")
    st.write("""
### Graph Neural Network (GNN)
Learns similarity patterns between movies using shared metadata / relationships.

### Knowledge Graph Convolutional Network (KGCN)
Learns entity-level relationships: genres, directors, companies, etc.

### XGBoost
150+ engineered features capturing metadata, engagement, and representation.
""")

    st.header("Stacking Ensemble")
    meta = artifacts["ensemble_meta"]
    rmse = meta.get("stacking_rmse", 0.1085)

    cols = st.columns(2)
    cols[0].metric("Stacking RMSE", f"{rmse:.4f}")
    cols[1].metric("Within ±10%", "80.2%")


# -----------------------------------------------------------------------------
# VISUAL GRAPH PAGE
# -----------------------------------------------------------------------------

def page_graph(db):
    st.title("Visual Graph Explorer")

    titles = all_titles(db)
    title = st.selectbox("Select a movie:", ["-- Select --"] + titles)

    if title != "-- Select --":
        movie = search_movie(db, title)
        fig = build_graph(db, movie)
        if fig:
            st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# ACKNOWLEDGEMENTS
# -----------------------------------------------------------------------------

def page_ack():
    st.title("Acknowledgements")
    st.write("""
Team 15 — Cinemaniacs  
- Angela Cottone  
- Nidhi Deshmukh  
- Dylan Sidhu  
- Matthew Ward  
- Clara Wei  

Data sources: TMDB, Rotten Tomatoes, YouTube API.

Thanks to the STA 160 instructional staff and tools used throughout development.
""")


# -----------------------------------------------------------------------------
# MAIN CONTROLLER
# -----------------------------------------------------------------------------

def main():
    db = get_db()
    artifacts = load_artifacts()

    # SIDEBAR NAVIGATION
    nav = st.sidebar.radio(
        "Navigation",
        ["Home", "Movie Search", "Compare Movies",
         "Analytics Dashboard", "Modeling",
         "Visual Graph Explorer", "Acknowledgements"],
        index=["Home", "Movie Search", "Compare Movies",
               "Analytics Dashboard", "Modeling",
               "Visual Graph Explorer", "Acknowledgements"].index(st.session_state["nav"])
    )

    st.session_state["nav"] = nav

    # ROUTING
    if nav == "Home":
        page_home(db)
    elif nav == "Movie Search":
        page_movie_search(db, artifacts)
    elif nav == "Compare Movies":
        page_compare(db, artifacts)
    elif nav == "Analytics Dashboard":
        page_analytics(db)
    elif nav == "Modeling":
        page_modeling(artifacts)
    elif nav == "Visual Graph Explorer":
        page_graph(db)
    else:
        page_ack()


if __name__ == "__main__":
    main()
