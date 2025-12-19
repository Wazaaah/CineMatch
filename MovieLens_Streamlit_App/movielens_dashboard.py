import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="CineMatch Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# PROFESSIONAL CSS WITH BEAUTIFUL TYPOGRAPHY - LARGER FONTS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700;800;900&family=Source+Sans+Pro:wght@300;400;600;700&display=swap');
    
    /* Main app background - Elegant gradient */
    .stApp {
        background: linear-gradient(165deg, #0a0e27 0%, #1a1f3a 40%, #0f1729 100%);
        font-family: 'Source Sans Pro', sans-serif;
    }
    
    /* Main content */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1800px;
    }
    
    /* Headers - Serif elegance */
    h1 {
        font-family: 'Playfair Display', serif !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 5.5rem !important;
        margin-bottom: 0.5rem !important;
        letter-spacing: -0.5px !important;
        line-height: 1.1 !important;
    }
    
    h2 {
        font-family: 'Playfair Display', serif !important;
        color: #e8e6e3 !important;
        font-weight: 600 !important;
        font-size: 3.6rem !important;
        margin-top: 3rem !important;
        margin-bottom: 1.5rem !important;
        letter-spacing: -0.3px !important;
    }
    
    h3 {
        font-family: 'Playfair Display', serif !important;
        color: #d4d2cf !important;
        font-weight: 600 !important;
        font-size: 2.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    h4 {
        font-family: 'Source Sans Pro', sans-serif !important;
        color: #c1bfbc !important;
        font-weight: 600 !important;
        font-size: 2.2rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
    }
    
    /* Body text - Clean sans-serif - LARGER */
    p, .stMarkdown, li {
        font-family: 'Source Sans Pro', sans-serif !important;
        color: #b8b6b3 !important;
        font-size: 2.2rem !important;
        line-height: 1.7 !important;
        font-weight: 400 !important;
    }
    
    /* Sidebar - Refined */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #151a30 0%, #0a0e27 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #e8e6e3 !important;
    }
    
    [data-testid="stSidebar"] .stRadio label {
        font-size: 1.6rem !important;
        font-weight: 500 !important;
        color: #d4d2cf !important;
        padding: 0.6rem 0 !important;
    }
    
    /* Metrics - Elegant cards - LARGER */
    [data-testid="stMetricValue"] {
        font-family: 'Playfair Display', serif !important;
        font-size: 5.2rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        letter-spacing: -0.5px !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-family: 'Source Sans Pro', sans-serif !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #8e8c89 !important;
        text-transform: uppercase !important;
        letter-spacing: 1.2px !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 1.4rem !important;
        font-weight: 500 !important;
    }
    
    /* Metric containers */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.03) 0%, rgba(255, 255, 255, 0.01) 100%);
        border-radius: 12px;
        padding: 1.8rem 1.5rem !important;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        border-color: rgba(255, 255, 255, 0.15);
        box-shadow: 0 6px 28px rgba(0, 0, 0, 0.4);
        transform: translateY(-2px);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        font-size: 1.5rem;
        font-family: 'Source Sans Pro', sans-serif;
        transition: all 0.3s ease;
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.15) 0%, rgba(255, 255, 255, 0.08) 100%);
        border-color: rgba(255, 255, 255, 0.3);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        margin: 2.5rem 0;
    }
    
    /* Info boxes - LARGER */
    .stAlert {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.08) 0%, rgba(59, 130, 246, 0.04) 100%);
        border-left: 3px solid rgba(59, 130, 246, 0.5);
        border-radius: 8px;
        padding: 1.2rem;
        font-size: 1.6rem !important;
        color: #b8b6b3 !important;
    }
    
    /* Selectbox - LARGER */
    .stSelectbox label {
        font-size: 1.6rem !important;
        font-weight: 600 !important;
        color: #d4d2cf !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        font-size: 1.5rem !important;
        color: #ffffff !important;
    }
    
    .stSelectbox > div > div > div {
        color: #ffffff !important;
    }
    
    ul[role="listbox"] {
        background-color: #151a30 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }

    ul[role="listbox"] li {
        color: #d4d2cf !important;
        font-size: 1.5rem !important;
    }
    
    /* Dataframe */
    .stDataFrame {
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Charts */
    .js-plotly-plot {
        border-radius: 8px;
    }
    
    /* Custom insight box */
    .insight-box {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0.02) 100%);
        border-left: 3px solid rgba(255, 255, 255, 0.3);
        border-radius: 8px;
        padding: 1.8rem;
        margin: 2rem 0;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .insight-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.1rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 1rem;
    }
    
    .insight-content {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 1.3rem;
        line-height: 1.7;
        color: #b8b6b3;
        margin-bottom: 1.2rem;
    }
    
    .insight-recommendation {
        background: rgba(34, 197, 94, 0.08);
        border-left: 2px solid rgba(34, 197, 94, 0.5);
        padding: 1rem 1.2rem;
        border-radius: 6px;
        margin-bottom: 0.8rem;
    }
    
    .insight-impact {
        font-size: 1.5rem;
        color: #fbbf24;
        font-weight: 500;
    }
    
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).parent
# Load data
@st.cache_data
def load_data():
    try:
        return {
            "users": pd.read_csv("./powerbi_user_stats.csv", parse_dates=["first_rating_date", "last_rating_date"]),
            "movies": pd.read_csv("./powerbi_movie_stats.csv"),
            "genres": pd.read_csv("./powerbi_genre_analysis.csv"),
            "trends": pd.read_csv("./powerbi_monthly_trends.csv", parse_dates=["date"]),
            "tags": pd.read_csv("./powerbi_tag_stats.csv"),
            "tag_sentiment": pd.read_csv("./powerbi_tag_sentiment.csv"),
            "years": pd.read_csv("./powerbi_year_performance.csv"),
            "ratings": pd.read_csv("./powerbi_ratings_sample.csv"),
        }
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

data = load_data()
if data is None:
    st.stop()

user_df = data["users"]
movie_df = data["movies"]
genre_df = data["genres"]
trend_df = data["trends"]
tag_sent_df = data["tag_sentiment"]
year_df = data["years"]
ratings_df = data["ratings"]

# Global KPIs - Hardcoded from notebook for accuracy
TOTAL_USERS = 330_975
TOTAL_RATINGS = 33_832_162
AVG_RATING = 3.70
TOTAL_MOVIES = 83_239
TOTAL_TAGS = 143_263
HARSH_PCT = 11.2
NEUTRAL_PCT = 60.8
GENEROUS_PCT = 28.0

# Calculate from data
total_movies = len(movie_df)
hidden_gems = len(movie_df[movie_df['hidden_gem'] == 1])
active_genres = len(genre_df)
power_users_count = len(user_df[user_df['engagement_level'] == 'Power User'])

# Helper function
def insight_box(title, insight, recommendation, impact=None):
    impact_html = f"<div class='insight-impact'><strong>Impact:</strong> {impact}</div>" if impact else ""
    st.markdown(f"""
    <div class='insight-box'>
        <div class='insight-title'>{title}</div>
        <div class='insight-content'><strong>Insight:</strong> {insight}</div>
        <div class='insight-recommendation'>
            <strong>Recommendation:</strong> {recommendation}
        </div>
        {impact_html}
    </div>
    """, unsafe_allow_html=True)

# SIDEBAR
st.sidebar.markdown("# CineMatch")
st.sidebar.markdown("### Analytics Platform")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Executive Overview", "User Analytics", "Content Performance", "Advanced Insights"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style='background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(59, 130, 246, 0.05) 100%);
            border-radius: 8px;
            padding: 1.5rem;
            border: 1px solid rgba(59, 130, 246, 0.2);
            margin-top: 2rem;'>
    <p style='color: #93c5fd; margin: 0; font-weight: 600; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 1px;'>Machine Learning Platform</p>
    <p style='color: #dbeafe; margin: 0.6rem 0 0 0; font-size: 1.2rem; font-weight: 400;'>8 models evaluated<br>33.8M ratings analyzed<br>330K+ users profiled</p>
</div>
""", unsafe_allow_html=True)

# HERO SECTION
st.markdown("""
<div style='background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.03) 100%);
            border-radius: 12px;
            padding: 3.5rem 3rem;
            margin-bottom: 3rem;
            border: 1px solid rgba(255, 255, 255, 0.1);'>
    <h1 style='margin-bottom: 0.8rem !important;'>CineMatch Platform</h1>
    <p style='font-size: 1.5rem !important; line-height: 1.6 !important; color: #b8b6b3 !important; margin: 0; font-weight: 300;'>
        Advanced machine learning analytics transforming viewer behavior data into strategic business insights.<br>
        <span style='font-size: 1.25rem; color: #8e8c89;'>Predictive modeling • User segmentation • Content optimization • Revenue intelligence</span>
    </p>
</div>
""", unsafe_allow_html=True)

# EXECUTIVE OVERVIEW PAGE
if page == "Executive Overview":
    st.markdown("## Executive Overview")
    st.markdown("Platform health metrics and strategic performance indicators")
    
    st.markdown("---")
    
    # Filter section
    filter_col, _ = st.columns([1, 3])
    with filter_col:
        selected_rater_type = st.selectbox(
            "Filter by Rater Type",
            ['All Users', 'Harsh', 'Neutral', 'Generous']
        )
    
    st.markdown("---")

    # Filter data
    if selected_rater_type == 'All Users':
        filtered_user_df = user_df
        display_users = TOTAL_USERS
        display_ratings = TOTAL_RATINGS
        display_avg = AVG_RATING
    else:
        filtered_user_df = user_df[user_df['rater_type'] == selected_rater_type]
        display_users = len(filtered_user_df)
        display_ratings = filtered_user_df['total_ratings'].sum()
        display_avg = filtered_user_df['avg_rating_given'].mean()
    
    power_users = len(filtered_user_df[filtered_user_df['engagement_level'] == 'Power User'])
    
    # KPI Row 1
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Users", f"{display_users:,}")
    
    with col2:
        st.metric("Platform Ratings", f"{display_ratings/1_000_000:.1f}M")
    
    with col3:
        st.metric("Movie Catalog", f"{TOTAL_MOVIES:,}")
    
    with col4:
        st.metric("Average Rating", f"{display_avg:.2f}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # KPI Row 2
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Power Users", f"{power_users:,}")
    
    with col2:
        st.metric("Hidden Gems", f"{hidden_gems}")
    
    with col3:
        st.metric("Active Genres", f"{active_genres}")
    
    with col4:
        st.metric("Unique Tags", f"{TOTAL_TAGS:,}")
    
    st.markdown("---")
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Platform Growth Trajectory")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=trend_df['date'],
            y=trend_df['rating_count'],
            name="Rating Volume",
            line=dict(color='rgba(59, 130, 246, 0.8)', width=3),
            fill='tonexty',
            fillcolor='rgba(59, 130, 246, 0.1)'
        ))
        
        fig.update_layout(
            template="plotly_dark",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(size=20, color='#b8b6b3', family='Source Sans Pro'),
            xaxis=dict(showgrid=False, title=""),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Rating Count"),
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Quality Consistency")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=trend_df['date'],
            y=trend_df['avg_rating'],
            name="Average Rating",
            line=dict(color='rgba(34, 197, 94, 0.8)', width=3),
            fill='tonexty',
            fillcolor='rgba(34, 197, 94, 0.1)'
        ))
        
        fig.update_layout(
            template="plotly_dark",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(size=20, color='#b8b6b3', family='Source Sans Pro'),
            xaxis=dict(showgrid=False, title=""),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Average Rating", range=[3.0, 4.0]),
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Strategic Insights
    st.markdown("### Strategic Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        insight_box(
            title="Stable User Base Composition",
            insight=f"{NEUTRAL_PCT}% of users are Neutral raters (average rating 3.0-4.0), representing the platform's stable core audience with balanced rating behavior.",
            recommendation="Develop tiered recommendation algorithms that account for rating tendencies. Implement user-type-specific personalization to improve engagement across all segments.",
            impact="High - Potential 15-20% increase in user satisfaction and retention"
        )
    
    with col2:
        insight_box(
            title="Untapped Content Value",
            insight=f"{hidden_gems} hidden gems identified with strong quality metrics (≥4.0 rating) but limited visibility (100-1000 ratings). These represent significant untapped catalog value.",
            recommendation="Launch targeted 'Hidden Gems' discovery feature integrated into recommendation engine. Create curated collections and promotional campaigns.",
            impact="Medium-High - Could increase catalog utilization by 12-18% and reduce content acquisition costs"
        )
    
    st.markdown("---")
    
    # Rating distribution
    st.markdown("### Rating Distribution Analysis")
    
    rating_counts = ratings_df['rating'].value_counts().sort_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=rating_counts.index,
        y=rating_counts.values,
        marker=dict(
            color=rating_counts.values,
            colorscale='Blues',
            showscale=False
        ),
        text=rating_counts.values,
        texttemplate='%{text:,}',
        textposition='outside',
        textfont=dict(size=20)
    ))
    
    fig.update_layout(
        template="plotly_dark",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=20, color='#b8b6b3', family='Source Sans Pro'),
        xaxis=dict(showgrid=False, title="Rating Value"),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Frequency"),
        margin=dict(l=20, r=20, t=20, b=40),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info(f"Platform demonstrates healthy rating distribution with strong skew toward positive ratings (4.0-5.0). Average rating of {AVG_RATING} indicates overall user satisfaction.")

# USER ANALYTICS PAGE
elif page == "User Analytics":
    st.markdown("## User Analytics")
    st.markdown("Behavioral segmentation and engagement patterns")
    st.markdown("---")
    
    # Key stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        harsh_count = int(TOTAL_USERS * HARSH_PCT / 100)
        st.metric("Harsh Raters", f"{harsh_count:,}", f"{HARSH_PCT}%")
    
    with col2:
        neutral_count = int(TOTAL_USERS * NEUTRAL_PCT / 100)
        st.metric("Neutral Raters", f"{neutral_count:,}", f"{NEUTRAL_PCT}%")
    
    with col3:
        generous_count = int(TOTAL_USERS * GENEROUS_PCT / 100)
        st.metric("Generous Raters", f"{generous_count:,}", f"{GENEROUS_PCT}%")
    
    with col4:
        st.metric("Avg Ratings Per User", "102")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Rater Type Distribution")
        st.markdown("""
        <div style='font-size: 1.1rem; color: #8e8c89; margin-bottom: 1.5rem;'>
        <strong>Classification Criteria:</strong><br>
        Harsh: 0-3.0 avg rating | Neutral: 3.0-4.0 | Generous: 4.0-5.0
        </div>
        """, unsafe_allow_html=True)
        
        rater_dist = user_df['rater_type'].value_counts().reset_index()
        rater_dist.columns = ['type', 'count']
        
        fig = go.Figure(data=[go.Pie(
            labels=rater_dist['type'],
            values=rater_dist['count'],
            hole=0.5,
            marker=dict(colors=['#ef4444', '#3b82f6', '#22c55e']),
            textfont=dict(size=20, family='Source Sans Pro')
        )])
        
        fig.update_layout(
            template="plotly_dark",
            height=450,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(size=20, color='#b8b6b3', family='Source Sans Pro'),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Characteristics table
        st.markdown("#### Rating Behavior Characteristics")
        
        characteristics_data = {
            'Rater Type': ['Harsh', 'Neutral', 'Generous'],
            'Avg Rating': [2.59, 3.59, 4.37],
            'Avg Ratings Given': [115, 120, 59],
            'Rating Variability': [1.19, 0.96, 0.69]
        }
        
        st.dataframe(
            pd.DataFrame(characteristics_data).style.format({
                'Avg Rating': '{:.2f}',
                'Avg Ratings Given': '{:.0f}',
                'Rating Variability': '{:.2f}'
            }),
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.markdown("### Engagement Level Distribution")
        st.markdown("""
        <div style='font-size: 1.1rem; color: #8e8c89; margin-bottom: 1.5rem;'>
        <strong>Classification Criteria:</strong><br>
        Casual: 0-20 | Regular: 20-100 | Active: 100-500 | Power User: 500+ ratings
        </div>
        """, unsafe_allow_html=True)
        
        engagement_dist = user_df['engagement_level'].value_counts().reset_index()
        engagement_dist.columns = ['level', 'users']
        
        # Order correctly
        order = ['Casual', 'Regular', 'Active', 'Power User']
        engagement_dist['level'] = pd.Categorical(engagement_dist['level'], categories=order, ordered=True)
        engagement_dist = engagement_dist.sort_values('level')
        
        fig = go.Figure()
        
        colors = ['#6b7280', '#3b82f6', '#8b5cf6', '#ef4444']
        
        fig.add_trace(go.Bar(
            x=engagement_dist['users'],
            y=engagement_dist['level'],
            orientation='h',
            marker=dict(color=colors),
            text=engagement_dist['users'],
            texttemplate='%{text:,}',
            textposition='outside',
            textfont=dict(size=20)
        ))
        
        fig.update_layout(
            template="plotly_dark",
            height=450,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(size=16, color='#b8b6b3', family='Source Sans Pro'),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Number of Users"),
            yaxis=dict(showgrid=False, title=""),
            margin=dict(l=20, r=80, t=20, b=40),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        power_user_pct = len(user_df[user_df['engagement_level'] == 'Power User']) / len(user_df) * 100
        st.info(f"Power users represent {power_user_pct:.1f}% of the user base but contribute disproportionately to platform engagement with 500+ ratings each.")
    
    st.markdown("---")




    # User activity distribution
    st.markdown("### User Activity Distribution")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Full Distribution")
        
        # Border
        st.markdown("""
        <div style='border: 2px solid rgba(59, 130, 246, 0.5); border-radius: 8px; padding: 1rem;'>
        """, unsafe_allow_html=True)
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=user_df['total_ratings'],
            nbinsx=100,
            marker=dict(
                color='rgba(59, 130, 246, 0.7)', 
                line=dict(color='rgba(59, 130, 246, 1)', width=1)
            )
        ))
        
        fig.update_layout(
            template="plotly_dark",
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30, 41, 59, 0.4)',
            font=dict(size=18, color='#b8b6b3', family='Source Sans Pro'),
            xaxis=dict(
                showgrid=True, 
                gridcolor='rgba(255,255,255,0.08)',
                title="Total Ratings per User",
                titlefont=dict(size=18),
                tickfont=dict(size=16)
            ),
            yaxis=dict(
                showgrid=True, 
                gridcolor='rgba(255,255,255,0.08)',
                title="Number of Users",
                titlefont=dict(size=18),
                tickfont=dict(size=16),
                tickformat=','
            ),
            margin=dict(l=80, r=20, t=20, b=60),
            showlegend=False
        )
        
        # Borders
        fig.update_xaxes(showline=True, linewidth=3, linecolor='rgba(59, 130, 246, 0.7)', mirror=True)
        fig.update_yaxes(showline=True, linewidth=3, linecolor='rgba(59, 130, 246, 0.7)', mirror=True)
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption("Shows all users (0-30k+ ratings)")

    with col2:
        st.markdown("#### Detailed View (0-500 ratings)")
        
        # Border
        st.markdown("""
        <div style='border: 2px solid rgba(239, 68, 68, 0.5); border-radius: 8px; padding: 1rem;'>
        """, unsafe_allow_html=True)
        
        # Filter to 0-500 ratings
        filtered = user_df[user_df['total_ratings'] <= 500]
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=filtered['total_ratings'],
            xbins=dict(start=0, end=500, size=10),
            marker=dict(
                color='rgba(239, 68, 68, 0.7)', 
                line=dict(color='rgba(255, 255, 255, 0.6)', width=1)
            )
        ))
        
        fig.update_layout(
            template="plotly_dark",
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30, 41, 59, 0.4)',
            font=dict(size=18, color='#b8b6b3', family='Source Sans Pro'),
            bargap=0.03,
            xaxis=dict(
                showgrid=True, 
                gridcolor='rgba(255,255,255,0.08)',
                title="Total Ratings per User",
                titlefont=dict(size=18),
                tickfont=dict(size=16),
                range=[0, 500]
            ),
            yaxis=dict(
                showgrid=True, 
                gridcolor='rgba(255,255,255,0.08)',
                title="Number of Users",
                titlefont=dict(size=18),
                tickfont=dict(size=16),
                tickformat=','
            ),
            margin=dict(l=80, r=20, t=20, b=60),
            showlegend=False
        )
        
        # Borders
        fig.update_xaxes(showline=True, linewidth=3, linecolor='rgba(239, 68, 68, 0.7)', mirror=True)
        fig.update_yaxes(showline=True, linewidth=3, linecolor='rgba(239, 68, 68, 0.7)', mirror=True)
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption(f"Zoomed view: {len(filtered):,} users ({len(filtered)/len(user_df)*100:.1f}%)")



    
    st.markdown("---")
    
    # Insights
    insight_box(
        title="Power User Retention Strategy",
        insight=f"Power Users ({power_users_count:,} users with 500+ ratings) demonstrate 3x longer platform tenure and consistent engagement patterns. This segment represents the highest-value users for long-term platform health.",
        recommendation="Implement VIP contributor program with benefits: early access to new releases, exclusive screenings, enhanced recommendation features, and community recognition. Create feedback loops to leverage their expertise.",
        impact="Very High - Estimated 25-30% reduction in churn for high-value segment, potential revenue uplift of 15-20%"
    )

# CONTENT PERFORMANCE PAGE
elif page == "Content Performance":
    st.markdown("## Content Performance")
    st.markdown("Genre analytics and catalog optimization insights")
    st.markdown("---")
    
    # Genre performance scatter
    st.markdown("### Genre Performance Matrix")
    st.markdown("Quality vs Popularity positioning of genre categories")
    
    fig = go.Figure()
    
    for idx, row in genre_df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['total_ratings']],
            y=[row['avg_rating']],
            mode='markers+text',
            name=row['genre'],
            marker=dict(
                size=np.sqrt(row['unique_users']) / 5,
                color=row['avg_rating'],
                colorscale='Viridis',
                showscale=False,
                line=dict(color='white', width=1)
            ),
            text=row['genre'],
            textposition="top center",
            textfont=dict(size=16),
            showlegend=False
        ))
    
    fig.update_layout(
        template="plotly_dark",
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=16, color='#b8b6b3', family='Source Sans Pro'),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Total Ratings (Popularity)", type='log'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Average Rating (Quality)"),
        margin=dict(l=40, r=20, t=20, b=40),
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("Bubble size represents unique users. Top-right quadrant genres (high quality, high popularity) represent optimal content acquisition targets.")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Top Genres by Quality")
        
        top_quality = genre_df.nlargest(10, 'avg_rating')
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=top_quality['avg_rating'],
            y=top_quality['genre'],
            orientation='h',
            marker=dict(
                color=top_quality['avg_rating'],
                colorscale='Teal',
                showscale=False
            ),
            text=top_quality['avg_rating'].round(3),
            texttemplate='%{text}',
            textposition='outside',
            textfont=dict(size=19)
        ))
        
        fig.update_layout(
            template="plotly_dark",
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(size=16, color='#b8b6b3', family='Source Sans Pro'),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Average Rating"),
            yaxis=dict(showgrid=False, title=""),
            margin=dict(l=20, r=80, t=20, b=40),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Top Genres by Volume")
        
        top_volume = genre_df.nlargest(10, 'total_ratings')
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=top_volume['total_ratings'],
            y=top_volume['genre'],
            orientation='h',
            marker=dict(
                color=top_volume['total_ratings'],
                colorscale='Purp',
                showscale=False
            ),
            text=top_volume['total_ratings'],
            texttemplate='%{text:,}',
            textposition='outside',
            textfont=dict(size=19)
        ))
        
        fig.update_layout(
            template="plotly_dark",
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(size=20, color='#b8b6b3', family='Source Sans Pro'),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Total Ratings"),
            yaxis=dict(showgrid=False, title=""),
            margin=dict(l=20, r=100, t=20, b=40),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Genre insights
    col1, col2 = st.columns(2)
    
    with col1:
        top_genre = genre_df.iloc[0]
        insight_box(
            title="Drama Category Dominance",
            insight=f"Drama leads across all metrics with {top_genre['total_ratings']:,} ratings and {top_genre['avg_rating']:.3f} average quality score. Represents optimal balance of volume and quality.",
            recommendation="Prioritize Drama content acquisition with 15-20% budget allocation increase. Focus on award-winning titles and critically acclaimed releases.",
            impact="High - Direct revenue impact through increased user engagement and retention"
        )
    
    with col2:
        insight_box(
            title="Film-Noir Quality Premium",
            insight="Film-Noir achieves highest quality rating (3.928) despite lower volume, indicating strong niche appeal and dedicated audience.",
            recommendation="Develop curated Film-Noir collections and targeted marketing campaigns. Expand catalog with classic and modern noir titles.",
            impact="Medium - Could unlock high-value niche audience segment with lower customer acquisition costs"
        )
    
    st.markdown("---")
    
    # Tag sentiment analysis
    st.markdown("### Tag Sentiment Performance")
    
    top_tags = tag_sent_df.nlargest(20, 'avg_rating')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=top_tags['avg_rating'],
        y=top_tags['tag'],
        orientation='h',
        marker=dict(
            color=top_tags['avg_rating'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Rating")
        ),
        text=top_tags['avg_rating'].round(2),
        texttemplate='%{text}',
        textposition='outside',
        textfont=dict(size=18)
    ))
    
    fig.update_layout(
        template="plotly_dark",
        height=700,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=20, color='#b8b6b3', family='Source Sans Pro'),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Average Rating"),
        yaxis=dict(showgrid=False, title=""),
        margin=dict(l=20, r=80, t=20, b=40)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info(f"Analysis of {len(tag_sent_df):,} unique tags reveals strong correlation between positive perception tags and higher ratings. Tags associated with quality storytelling and production values drive viewer satisfaction.")

# ADVANCED INSIGHTS PAGE
elif page == "Advanced Insights":
    st.markdown("## Advanced Insights")
    st.markdown("Machine learning discoveries and hidden patterns")
    st.markdown("---")
    
    # Key discoveries
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(139, 92, 246, 0.12) 0%, rgba(139, 92, 246, 0.05) 100%);
                    border-radius: 10px;
                    padding: 2rem;
                    text-align: center;
                    border: 1px solid rgba(139, 92, 246, 0.2);'>
            <h4 style='color: #c4b5fd; margin-bottom: 1rem; font-size: 1.2rem;'>TEMPORAL EFFECT</h4>
            <p style='color: #e9d5ff; margin-bottom: 1rem; font-size: 1.15rem;'>
                Older films rated higher
            </p>
            <div style='font-size: 3.5rem; font-weight: 700; color: #ffffff; font-family: "Playfair Display", serif;'>+0.15</div>
            <p style='color: #c4b5fd; margin-top: 0.5rem; font-size: 1.05rem;'>rating points</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, rgba(34, 197, 94, 0.12) 0%, rgba(34, 197, 94, 0.05) 100%);
                    border-radius: 10px;
                    padding: 2rem;
                    text-align: center;
                    border: 1px solid rgba(34, 197, 94, 0.2);'>
            <h4 style='color: #86efac; margin-bottom: 1rem; font-size: 1.2rem;'>HIDDEN GEMS</h4>
            <p style='color: #bbf7d0; margin-bottom: 1rem; font-size: 1.15rem;'>
                High quality, limited visibility
            </p>
            <div style='font-size: 3.5rem; font-weight: 700; color: #ffffff; font-family: "Playfair Display", serif;'>{hidden_gems}</div>
            <p style='color: #86efac; margin-top: 0.5rem; font-size: 1.05rem;'>100-1000 ratings, ≥4.0 avg</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(59, 130, 246, 0.12) 0%, rgba(59, 130, 246, 0.05) 100%);
                    border-radius: 10px;
                    padding: 2rem;
                    text-align: center;
                    border: 1px solid rgba(59, 130, 246, 0.2);'>
            <h4 style='color: #93c5fd; margin-bottom: 1rem; font-size: 1.2rem;'>USER CLUSTERS</h4>
            <p style='color: #dbeafe; margin-bottom: 1rem; font-size: 1.15rem;'>
                Distinct taste profiles
            </p>
            <div style='font-size: 3.5rem; font-weight: 700; color: #ffffff; font-family: "Playfair Display", serif;'>5</div>
            <p style='color: #93c5fd; margin-top: 0.5rem; font-size: 1.05rem;'>K-means clusters</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    insight_box(
        title="Machine Learning Clustering Analysis",
        insight="K-means clustering (k=5) identified five distinct user taste profiles with clear genre preferences and rating patterns. Clusters range from mainstream blockbuster enthusiasts to art-house specialists.",
        recommendation="Implement cluster-based recommendation system with personalized homepage experiences. Develop targeted content acquisition strategy aligned with cluster preferences. Create segment-specific marketing campaigns.",
        impact="Very High - Expected 30-40% improvement in recommendation accuracy and 20-25% increase in user engagement"
    )
    
    st.markdown("---")
    
    # Release year trends
    st.markdown("### Temporal Rating Patterns")
    
    # Filter to relevant years
    year_filtered = year_df[year_df['release_year'] >= 1950].copy()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=year_filtered['release_year'],
        y=year_filtered['avg_rating'],
        mode='lines+markers',
        line=dict(color='rgba(139, 92, 246, 0.8)', width=3),
        marker=dict(size=8, color='rgba(139, 92, 246, 1)'),
        fill='tonexty',
        fillcolor='rgba(139, 92, 246, 0.1)'
    ))
    
    fig.update_layout(
        template="plotly_dark",
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=16, color='#b8b6b3', family='Source Sans Pro'),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Release Year"),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Average Rating"),
        margin=dict(l=40, r=20, t=20, b=40),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("Classic films (pre-1990) demonstrate systematically higher ratings, suggesting survivorship bias and nostalgic value. Consider this temporal effect in recommendation algorithms.")
    
    st.markdown("---")
    
    # Hidden gems showcase
    st.markdown("### Hidden Gems Catalog")
    st.markdown("High-quality titles with growth potential")
    
    hidden_gems_df = movie_df[movie_df['hidden_gem'] == 1].nlargest(20, 'avg_rating')
    
    display_df = hidden_gems_df[['title', 'genres', 'release_year', 'num_ratings', 'avg_rating']].copy()
    display_df.columns = ['Title', 'Genres', 'Year', 'Ratings', 'Avg Rating']
    
    st.dataframe(
        display_df.style.format({
            'Avg Rating': '{:.2f}',
            'Ratings': '{:,}',
            'Year': '{:.0f}'
        }),
        use_container_width=True,
        height=600
    )
    
    st.markdown("---")
    
    # ML Model performance
    st.markdown("### Machine Learning Model Performance")
    
    model_data = {
        'Model': ['GradientBoost', 'Ridge Regression', 'XGBoost', 'CatBoost', 'Random Forest', 'LightGBM', 'Elastic Net', 'SVR'],
        'RMSE': [0.8544, 0.8635, 0.8671, 0.8707, 0.8745, 0.8783, 0.8821, 0.8967],
        'Type': ['Ensemble', 'Linear', 'Ensemble', 'Ensemble', 'Ensemble', 'Ensemble', 'Linear', 'Kernel']
    }
    
    model_df = pd.DataFrame(model_data)
    
    fig = go.Figure()
    
    colors = ['#22c55e' if i == 0 else '#3b82f6' if i < 3 else '#6b7280' for i in range(len(model_df))]
    
    fig.add_trace(go.Bar(
        x=model_df['Model'],
        y=model_df['RMSE'],
        marker=dict(color=colors),
        text=model_df['RMSE'].round(4),
        texttemplate='%{text}',
        textposition='outside',
        textfont=dict(size=19)
    ))
    
    fig.update_layout(
        template="plotly_dark",
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=16, color='#b8b6b3', family='Source Sans Pro'),
        xaxis=dict(showgrid=False, title="Model"),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="RMSE (Lower is Better)"),
        margin=dict(l=40, r=20, t=40, b=40),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("GradientBoost achieved best performance (RMSE: 0.8544) across 8 evaluated models. Ensemble methods consistently outperformed linear models, indicating complex non-linear relationships in rating behavior.")
    
    st.markdown("---")
    
    # Feature importance
    st.markdown("### Key Prediction Features")
    
    col1, col2, col3 = st.columns(3)
    
    features = [
        ("User Avg Rating", "0.342", "Historical rating tendency"),
        ("Movie Avg Rating", "0.289", "Content quality signal"),
        ("Genre Preferences", "0.187", "Taste alignment"),
        ("Release Year", "0.094", "Temporal effects"),
        ("Rating Count", "0.051", "Popularity signal"),
        ("User Activity", "0.037", "Engagement level")
    ]
    
    for i, col in enumerate([col1, col2, col3]):
        for j in range(2):
            idx = i * 2 + j
            if idx < len(features):
                name, importance, desc = features[idx]
                with col:
                    st.markdown(f"""
                    <div style='background: rgba(255, 255, 255, 0.03);
                                border-radius: 8px;
                                padding: 1.2rem;
                                margin-bottom: 1rem;
                                border: 1px solid rgba(255, 255, 255, 0.06);'>
                        <div style='font-size: 0.95rem; color: #8e8c89; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.3rem;'>{name}</div>
                        <div style='font-size: 2rem; font-weight: 700; color: #ffffff; font-family: "Playfair Display", serif; margin-bottom: 0.3rem;'>{importance}</div>
                        <div style='font-size: 1.05rem; color: #b8b6b3;'>{desc}</div>
                    </div>
                    """, unsafe_allow_html=True)

# FOOTER
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; opacity: 0.6;'>
    <p style='font-size: 1.05rem; margin: 0; color: #8e8c89;'>
        CineMatch Platform • Machine Learning Project • Group 2 Final Assessment © 2025
    </p>
</div>
""", unsafe_allow_html=True)
