import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="MovieLens Business Insights Dashboard",
    layout="wide"
)

st.title("🎬 MovieLens Business Insights & Visual Analytics Dashboard")
st.markdown(
    """
    This dashboard translates **machine learning outputs and statistical analysis**
    into **business-relevant insights** using interactive visual analytics.

    **Focus Areas**
    - User behavior & retention  
    - Content performance & sentiment  
    - Hidden patterns & audience segmentation
    """
)

# ==================================================
# LOAD DATA
# ==================================================
@st.cache_data
def load_data():
    return {
        "users": pd.read_csv("powerbi_user_stats.csv", parse_dates=["first_rating_date", "last_rating_date"]),
        "movies": pd.read_csv("powerbi_movie_stats.csv"),
        "genres": pd.read_csv("powerbi_genre_analysis.csv"),
        "trends": pd.read_csv("powerbi_monthly_trends.csv", parse_dates=["date"]),
        "tags": pd.read_csv("powerbi_tag_stats.csv"),
        "tag_sentiment": pd.read_csv("powerbi_tag_sentiment.csv"),
        "years": pd.read_csv("powerbi_year_performance.csv"),
        "ratings": pd.read_csv("powerbi_ratings_sample.csv"),
    }

data = load_data()

user_df = data["users"]
movie_df = data["movies"]
genre_df = data["genres"]
trend_df = data["trends"]
tag_sent_df = data["tag_sentiment"]
year_df = data["years"]
ratings_df = data["ratings"]

# ==================================================
# SECTION 1: USER BEHAVIOR INSIGHTS
# ==================================================
st.header("👤 User Behavior Insights")

# --------------------------------------------------
# Harsh vs Generous Raters
# --------------------------------------------------
st.subheader("Harsh vs Generous Raters")

rater_dist = (
    user_df
    .groupby("rater_type")
    .size()
    .reset_index(name="user_count")
)

col1, col2 = st.columns(2)

with col1:
    fig = px.pie(
        rater_dist,
        names="rater_type",
        values="user_count",
        hole=0.5,
        title="Distribution of Rater Types"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    avg_rating = (
        user_df
        .groupby("rater_type")["avg_rating_given"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        avg_rating,
        x="rater_type",
        y="avg_rating_given",
        text_auto=".2f",
        title="Average Rating Given by Rater Type"
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
    **Insight:**  
    Harsh raters apply stricter standards, while generous raters inflate averages.
    This has implications for **rating bias** and **recommendation calibration**.
    """
)

# --------------------------------------------------
# Rating Evolution Over Time
# --------------------------------------------------
st.subheader("Rating Evolution Over Time")

fig = px.line(
    trend_df,
    x="date",
    y=["avg_rating", "rating_count"],
    title="Cinema Popularity and Rating Quality Over Time"
)
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
    **Insight:**  
    While rating volume grows steadily (platform expansion),
    average ratings stabilize — suggesting maturing user evaluation behavior.
    """
)

# --------------------------------------------------
# Retention Analysis
# --------------------------------------------------
st.subheader("User Retention & Engagement")

fig = px.box(
    user_df,
    x="engagement_level",
    y="activity_days",
    title="User Retention by Engagement Level"
)
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
    **Insight:**  
    Power users remain active for significantly longer periods.
    Retention strategies should prioritize converting casual users into regular contributors.
    """
)

# ==================================================
# SECTION 2: CONTENT INSIGHTS
# ==================================================
st.header("🎥 Content Performance & Sentiment")

# --------------------------------------------------
# Genre Performance
# --------------------------------------------------
st.subheader("Genre Performance: Popularity vs Quality")

fig = px.scatter(
    genre_df,
    x="total_ratings",
    y="avg_rating",
    size="unique_users",
    color="genre",
    title="Genre Popularity vs Quality (Bubble Size = Reach)",
    hover_data=["unique_movies"]
)
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
    **Insight:**  
    High engagement does not always imply high quality.
    Niche genres often deliver stronger satisfaction with smaller audiences.
    """
)

# --------------------------------------------------
# Tag Sentiment
# --------------------------------------------------
st.subheader("Tag Sentiment vs Ratings")

fig = px.bar(
    tag_sent_df.head(20),
    x="tag",
    y="avg_rating",
    text_auto=".2f",
    title="Top Sentiment Tags by Average Rating"
)
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
    **Insight:**  
    Positive perception tags (e.g., *underrated*, *masterpiece*)
    strongly correlate with higher ratings, validating tags as sentiment signals.
    """
)

# ==================================================
# SECTION 3: HIDDEN PATTERNS
# ==================================================
st.header("🔍 Hidden Patterns & Advanced Analytics")

# --------------------------------------------------
# Release Year Impact
# --------------------------------------------------
st.subheader("Impact of Release Year on Ratings")

fig = px.line(
    year_df,
    x="release_year",
    y=["avg_rating", "total_ratings"],
    title="Movie Performance Across Release Years"
)
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
    **Insight:**  
    Older movies often achieve higher average ratings but lower visibility,
    reflecting nostalgia bias and survivorship effects.
    """
)

# --------------------------------------------------
# Hidden Gems
# --------------------------------------------------
st.subheader("Hidden Gems: High Quality, Low Visibility")

hidden_gems = movie_df[movie_df["hidden_gem"] == 1].sort_values(
    ["avg_rating", "num_ratings"], ascending=False
)

st.dataframe(hidden_gems.head(15))

st.markdown(
    """
    **Insight:**  
    These films represent **undervalued catalog assets**
    that could be promoted to increase engagement and discovery.
    """
)

# --------------------------------------------------
# REAL USER–GENRE CLUSTERING
# --------------------------------------------------
st.subheader("Genre Affinity Clusters (User Taste Profiles)")

# Explode genres
ratings_genres = ratings_df.copy()
ratings_genres["genres"] = ratings_genres["genres"].str.split("|")
ratings_genres = ratings_genres.explode("genres")
ratings_genres = ratings_genres[ratings_genres["genres"].notna()]

# Build user–genre matrix
user_genre_matrix = ratings_genres.pivot_table(
    index="userId",
    columns="genres",
    values="rating",
    aggfunc="mean",
    fill_value=0
)

# Scale and cluster
scaler = StandardScaler()
user_genre_scaled = scaler.fit_transform(user_genre_matrix)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(user_genre_scaled)

user_genre_matrix["Cluster"] = clusters
cluster_profile = user_genre_matrix.groupby("Cluster").mean()

fig = px.imshow(
    cluster_profile,
    aspect="auto",
    title="User Genre Affinity Clusters"
)
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
    **Insight:**  
    Distinct taste profiles emerge, revealing actionable audience segments.
    These clusters support **personalized recommendations** and **targeted marketing**.
    """
)

# ==================================================
# EXECUTIVE SUMMARY
# ==================================================
st.header("📌 Executive Takeaways")

st.markdown(
    """
    - User activity is highly skewed, with power users driving engagement  
    - Genre popularity does not always align with satisfaction  
    - Tags provide meaningful sentiment signals beyond numeric ratings  
    - Hidden gems represent growth opportunities for catalog utilization  
    - Clustering uncovers audience taste profiles for personalization  

    **This dashboard demonstrates how machine learning outputs
    can be transformed into decision-ready business intelligence.**
    """
)
