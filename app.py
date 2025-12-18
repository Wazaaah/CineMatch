"""
🎬 CineMatch - Netflix-Style Movie Recommender
Flask Backend with full interactivity
Version 3.0 - Multi-Model Predictions (GradientBoost, Stacking, Ridge)
Best Model RMSE: 0.8544 (GradientBoost)
"""

from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import joblib
import requests
import re
import os

app = Flask(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

TMDB_API_KEY = os.environ.get('TMDB_API_KEY', '06d6b87cbe24892316e0b14a4714e8f5')
TMDB_IMAGE_BASE = 'https://image.tmdb.org/t/p/w500'
OMDB_API_KEY = os.environ.get('OMDB_API_KEY', '6e29ad83')

# =============================================================================
# DATA LOADING
# =============================================================================

DATA = {}

def load_data():
    """Load all deployment files"""
    global DATA
    print("🔄 Loading data...")

    # =========================================================================
    # ML MODELS (3 Best Models)
    # =========================================================================

    # GradientBoost - BEST MODEL (RMSE: 0.8544)
    try:
        DATA['gb_model'] = joblib.load('models/gradientboost_model.pkl')
        print("✓ GradientBoost loaded (BEST - RMSE: 0.8544)")
    except:
        DATA['gb_model'] = None
        print("⚠ GradientBoost not found")

    # Ridge Regression
    try:
        DATA['ridge_model'] = joblib.load('models/ridge_model.pkl')
        print("✓ Ridge loaded")
    except:
        DATA['ridge_model'] = None
        print("⚠ Ridge not found")


    # =========================================================================
    # PREPROCESSING (Retrained)
    # =========================================================================

    try:
        DATA['scaler'] = joblib.load('preprocessing/scaler.pkl')
        print("✓ Scaler loaded")
    except:
        DATA['scaler'] = None

    try:
        DATA['genre_encoder'] = joblib.load('preprocessing/genre_encoder.pkl')
        print("✓ Genre encoder loaded")
    except:
        DATA['genre_encoder'] = None

    # Feature columns - CRITICAL for correct feature order
    try:
        DATA['feature_columns'] = joblib.load('preprocessing/feature_columns.pkl')
        print(f"✓ Feature columns loaded ({len(DATA['feature_columns'])} features)")
    except:
        DATA['feature_columns'] = None
        print("⚠ Feature columns not found - predictions may fail")

    # =========================================================================
    # DATA FILES
    # =========================================================================

    try:
        DATA['movies'] = joblib.load('data/movies_data.pkl')
        print("✓ Movies loaded")
    except:
        try:
            DATA['movies'] = pd.read_csv('data/movies_catalog.csv')
            print("✓ Movies CSV loaded")
        except:
            DATA['movies'] = None

    try:
        DATA['movie_popularity'] = joblib.load('data/weighted_popularity_baseline.pkl')
        print("✓ Popularity loaded")
    except:
        try:
            DATA['movie_popularity'] = pd.read_csv('data/top_10000_movies.csv')
            print("✓ Top movies CSV loaded")
        except:
            DATA['movie_popularity'] = None

    try:
        DATA['genome_features'] = joblib.load('data/genome_features.pkl')
        # Set movieId as index if it's a column
        if isinstance(DATA['genome_features'], pd.DataFrame):
            if 'movieId' in DATA['genome_features'].columns:
                DATA['genome_features'] = DATA['genome_features'].set_index('movieId')
        print("✓ Genome features loaded")
    except:
        DATA['genome_features'] = None

    # Movie statistics (for enhanced features)
    try:
        DATA['movie_statistics'] = joblib.load('data/movie_statistics.pkl')
        # Set movieId as index if it's a column
        if isinstance(DATA['movie_statistics'], pd.DataFrame):
            if 'movieId' in DATA['movie_statistics'].columns:
                DATA['movie_statistics'] = DATA['movie_statistics'].set_index('movieId')
        print("✓ Movie statistics loaded")
    except:
        DATA['movie_statistics'] = None

    # User features (for enhanced features) - we'll use defaults for movie-only predictions
    try:
        DATA['user_features'] = joblib.load('data/user_features.pkl')
        print("✓ User features loaded")
    except:
        DATA['user_features'] = None

    # Links for TMDB/IMDb IDs
    try:
        DATA['links'] = pd.read_csv('data/links.csv')
        print("✓ Links loaded")
    except:
        try:
            DATA['links'] = pd.read_csv('links.csv')
            print("✓ Links loaded from root")
        except:
            DATA['links'] = None

    print("✅ Loading complete!")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_year(title):
    """Extract year from title"""
    match = re.search(r'\((\d{4})\)', str(title))
    return int(match.group(1)) if match else None


def get_all_genres():
    """Get unique genres"""
    if DATA['movies'] is None:
        return []
    genres = set()
    for g in DATA['movies']['genres'].dropna():
        for genre in str(g).split('|'):
            if genre and genre != '(no genres listed)':
                genres.add(genre)
    return sorted(list(genres))


def build_features(movie_id):
    """
    Build feature vector for a movie using the new enhanced feature set.
    Must match the training feature columns exactly.
    """
    if DATA['movies'] is None or DATA['genre_encoder'] is None:
        return None

    movie = DATA['movies'][DATA['movies']['movieId'] == movie_id]
    if len(movie) == 0:
        return None

    movie = movie.iloc[0]
    feature_cols = DATA.get('feature_columns')

    if feature_cols is None:
        print("Warning: feature_columns not loaded")
        return None

    try:
        # Initialize feature dict
        features = {}

        # ----- GENRE FEATURES (one-hot encoded) -----
        # Check if genre_list column exists, otherwise parse from genres
        if 'genre_list' in movie.index and isinstance(movie['genre_list'], list):
            genres_list = [movie['genre_list']]
        else:
            genres_list = [str(movie['genres']).split('|')]

        genre_encoded = DATA['genre_encoder'].transform(genres_list)[0]
        genre_classes = DATA['genre_encoder'].classes_

        for i, genre in enumerate(genre_classes):
            features[genre] = genre_encoded[i]

        # ----- RELEASE YEAR -----
        if 'release_year' in movie.index and pd.notna(movie['release_year']) and movie['release_year'] > 0:
            year = int(movie['release_year'])
        else:
            year = extract_year(movie['title']) or 2000
        features['release_year'] = year

        # ----- GENOME FEATURES -----
        genome_cols = ['genome_mean', 'genome_std', 'genome_min', 'genome_max']
        genome_defaults = {'genome_mean': 0.4, 'genome_std': 0.15, 'genome_min': 0.0, 'genome_max': 1.0}

        if DATA['genome_features'] is not None:
            gf = DATA['genome_features']
            if movie_id in gf.index:
                g = gf.loc[movie_id]
                for col in genome_cols:
                    if col in g.index:
                        features[col] = float(g[col])
                    else:
                        features[col] = genome_defaults[col]
            else:
                for col in genome_cols:
                    features[col] = genome_defaults[col]
        else:
            for col in genome_cols:
                features[col] = genome_defaults[col]

        # ----- USER BEHAVIOR FEATURES (use median/default values) -----
        # Since we don't have a specific user, use population averages
        user_cols = ['user_rating_count', 'user_avg_rating', 'user_rating_std']
        user_defaults = {'user_rating_count': 100, 'user_avg_rating': 3.5, 'user_rating_std': 1.0}

        for col in user_cols:
            features[col] = user_defaults[col]

        # ----- MOVIE STATISTICS FEATURES -----
        movie_stat_cols = ['movie_rating_count', 'movie_avg_rating', 'movie_rating_std']
        movie_stat_defaults = {'movie_rating_count': 100, 'movie_avg_rating': 3.5, 'movie_rating_std': 1.0}

        if DATA['movie_statistics'] is not None:
            ms = DATA['movie_statistics']
            if movie_id in ms.index:
                m = ms.loc[movie_id]
                for col in movie_stat_cols:
                    if col in m.index:
                        features[col] = float(m[col])
                    else:
                        features[col] = movie_stat_defaults[col]
            else:
                for col in movie_stat_cols:
                    features[col] = movie_stat_defaults[col]
        else:
            for col in movie_stat_cols:
                features[col] = movie_stat_defaults[col]

        # ----- BUILD FINAL FEATURE VECTOR IN CORRECT ORDER -----
        feature_vector = []
        for col in feature_cols:
            if col in features:
                feature_vector.append(features[col])
            else:
                # Feature was dropped due to multicollinearity, skip it
                # Or it's a genre we don't have - default to 0
                feature_vector.append(0)

        return np.array(feature_vector, dtype=np.float64)

    except Exception as e:
        print(f"Error building features for movie {movie_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


def predict_all_models(movie_id):
    """Get predictions from all 3 models for a movie"""
    predictions = {}

    features = build_features(movie_id)
    if features is None:
        return predictions

    features = features.reshape(1, -1)

    # GradientBoost (no scaling needed for tree-based)
    if DATA['gb_model'] is not None:
        try:
            pred = float(DATA['gb_model'].predict(features)[0])
            predictions['gradientBoost'] = round(np.clip(pred, 0.5, 5.0), 2)
        except Exception as e:
            print(f"GB prediction error: {e}")

    # Ridge (needs scaling)
    if DATA['ridge_model'] is not None and DATA['scaler'] is not None:
        try:
            features_scaled = DATA['scaler'].transform(features)
            pred = float(DATA['ridge_model'].predict(features_scaled)[0])
            predictions['ridge'] = round(np.clip(pred, 0.5, 5.0), 2)
        except Exception as e:
            print(f"Ridge prediction error: {e}")

    return predictions


def movie_to_dict(row):
    """Convert movie row to dictionary with TMDB and IMDb IDs"""
    title = row.get('title', 'Unknown')
    movie_id = int(row.get('movieId', 0))

    # Get IDs
    tmdb_id = None
    imdb_id = None

    if DATA['links'] is not None:
        link = DATA['links'][DATA['links']['movieId'] == movie_id]
        if len(link) > 0:
            tid = link.iloc[0].get('tmdbId')
            if pd.notna(tid):
                tmdb_id = int(tid)

            iid = link.iloc[0].get('imdbId')
            if pd.notna(iid):
                raw_iid = str(iid).split('.')[0]
                imdb_id = f"tt{int(raw_iid):07d}"

    return {
        'movieId': movie_id,
        'title': title,
        'cleanTitle': str(title).split('(')[0].strip(),
        'year': extract_year(title),
        'genres': str(row.get('genres', '')).split('|'),
        'avgRating': round(float(row['avg_rating']), 2) if pd.notna(row.get('avg_rating')) else None,
        'numRatings': int(row['num_ratings']) if pd.notna(row.get('num_ratings')) else None,
        'weightedScore': round(float(row['weighted_score']), 2) if pd.notna(row.get('weighted_score')) else None,
        'similarity': round(float(row['similarity']) * 100, 1) if pd.notna(row.get('similarity')) else None,
        'tmdbId': tmdb_id,
        'imdbId': imdb_id
    }


# =============================================================================
# ROUTES - PAGES
# =============================================================================

@app.route('/')
def index():
    """Main page"""
    genres = get_all_genres()
    return render_template('index.html', genres=genres, tmdb_api_key=TMDB_API_KEY)


@app.route('/api/imdb-poster/<imdb_id>')
def api_imdb_poster(imdb_id):
    """Fetch poster from OMDB API"""
    if not OMDB_API_KEY:
        return jsonify({'poster': None}), 500

    try:
        r = requests.get(
            'https://www.omdbapi.com/',
            params={'i': imdb_id, 'apikey': OMDB_API_KEY},
            timeout=5
        )
        if r.status_code == 200:
            data = r.json()
            poster = data.get('Poster')
            if poster and poster != 'N/A':
                return jsonify({'poster': poster})
        return jsonify({'poster': None})
    except:
        return jsonify({'poster': None}), 500


# =============================================================================
# ROUTES - API
# =============================================================================

@app.route('/api/trending')
def api_trending():
    """Get trending movies"""
    if DATA['movie_popularity'] is None:
        return jsonify([])

    genre = request.args.get('genre', '')
    sort_by = request.args.get('sort', 'trending')
    min_year = request.args.get('min_year', type=int)
    max_year = request.args.get('max_year', type=int)
    limit = request.args.get('limit', 20, type=int)

    df = DATA['movie_popularity'].copy()

    if 'release_year' not in df.columns:
        df['release_year'] = df['title'].apply(extract_year)

    if genre:
        df = df[df['genres'].str.contains(genre, case=False, na=False)]
    if min_year:
        df = df[df['release_year'] >= min_year]
    if max_year:
        df = df[df['release_year'] <= max_year]

    df = df.sample(frac=1, random_state=np.random.randint(0, 10000))

    if sort_by == 'trending':
        df = df.sort_values('weighted_score', ascending=False)
    elif sort_by == 'trending_asc':
        df = df.sort_values('weighted_score', ascending=True)
    elif sort_by == 'az':
        df = df.sort_values('title', ascending=True)
    elif sort_by == 'za':
        df = df.sort_values('title', ascending=False)
    elif sort_by == 'newest':
        df = df.sort_values('release_year', ascending=False)
    elif sort_by == 'oldest':
        df = df.sort_values('release_year', ascending=True)

    movies = [movie_to_dict(row) for _, row in df.head(limit).iterrows()]
    return jsonify(movies)


@app.route('/api/search')
def api_search():
    """Search movies"""
    query = request.args.get('q', '')
    if not query or len(query) < 2 or DATA['movies'] is None:
        return jsonify([])

    df = DATA['movies'].copy()
    mask = df['title'].str.lower().str.contains(query.lower(), na=False)
    results = df[mask].head(30)

    if DATA['movie_popularity'] is not None:
        pop = DATA['movie_popularity']
        cols = ['movieId'] + [c for c in ['avg_rating', 'num_ratings', 'weighted_score'] if c in pop.columns]
        results = results.merge(pop[cols], on='movieId', how='left')

    movies = [movie_to_dict(row) for _, row in results.iterrows()]
    return jsonify(movies)


@app.route('/api/movie/<int:movie_id>')
def api_movie(movie_id):
    """Get single movie details with predictions from ALL 3 models"""
    if DATA['movies'] is None:
        return jsonify({'error': 'No data'}), 404

    movie = DATA['movies'][DATA['movies']['movieId'] == movie_id]
    if len(movie) == 0:
        return jsonify({'error': 'Not found'}), 404

    movie = movie.iloc[0].to_dict()

    # Add ratings
    if DATA['movie_popularity'] is not None:
        pop = DATA['movie_popularity'][DATA['movie_popularity']['movieId'] == movie_id]
        if len(pop) > 0:
            pop = pop.iloc[0]
            movie['avg_rating'] = pop.get('avg_rating')
            movie['num_ratings'] = pop.get('num_ratings')
            movie['weighted_score'] = pop.get('weighted_score')

    result = movie_to_dict(movie)

    # Get predictions from ALL 3 models
    predictions = predict_all_models(movie_id)
    result['predictions'] = predictions

    return jsonify(result)


@app.route('/api/similar/<int:movie_id>')
def api_similar(movie_id):
    """Get similar movies"""
    if DATA['movies'] is None:
        return jsonify({'movies': [], 'method': 'error'})

    limit = request.args.get('limit', 12, type=int)
    movies = DATA['movies']

    target = movies[movies['movieId'] == movie_id]
    if len(target) == 0:
        return jsonify({'movies': [], 'method': 'error'})

    method = 'genre'

    # Try feature-based similarity
    if DATA['genre_encoder'] is not None and DATA['scaler'] is not None:
        try:
            target_features = build_features(movie_id)
            if target_features is not None:
                if DATA['movie_popularity'] is not None:
                    candidates = DATA['movie_popularity'].head(1000)['movieId'].tolist()
                else:
                    candidates = movies.head(1000)['movieId'].tolist()

                candidates = [c for c in candidates if c != movie_id]

                similarities = []
                target_scaled = DATA['scaler'].transform(target_features.reshape(1, -1))[0]

                for cid in candidates[:300]:
                    feat = build_features(cid)
                    if feat is not None:
                        cand_scaled = DATA['scaler'].transform(feat.reshape(1, -1))[0]
                        dot = np.dot(target_scaled, cand_scaled)
                        norm_t = np.linalg.norm(target_scaled)
                        norm_c = np.linalg.norm(cand_scaled)
                        if norm_t > 0 and norm_c > 0:
                            sim = dot / (norm_t * norm_c)
                            similarities.append({'movieId': cid, 'similarity': sim})

                if similarities:
                    sim_df = pd.DataFrame(similarities).sort_values('similarity', ascending=False).head(limit)
                    sim_df = sim_df.merge(movies[['movieId', 'title', 'genres']], on='movieId', how='left')

                    if DATA['movie_popularity'] is not None:
                        pop = DATA['movie_popularity']
                        cols = ['movieId'] + [c for c in ['avg_rating', 'num_ratings', 'weighted_score'] if c in pop.columns]
                        sim_df = sim_df.merge(pop[cols], on='movieId', how='left')

                    result = [movie_to_dict(row) for _, row in sim_df.iterrows()]
                    return jsonify({'movies': result, 'method': 'features'})
        except Exception as e:
            print(f"Feature similarity error: {e}")

    # Fallback: genre-based
    target_genres = set(str(target.iloc[0]['genres']).split('|'))

    def genre_sim(g):
        mg = set(str(g).split('|'))
        inter = len(target_genres & mg)
        union = len(target_genres | mg)
        return inter / union if union > 0 else 0

    df = movies.copy()
    df['similarity'] = df['genres'].apply(genre_sim)
    df = df[df['movieId'] != movie_id].sort_values('similarity', ascending=False).head(limit)

    if DATA['movie_popularity'] is not None:
        pop = DATA['movie_popularity']
        cols = ['movieId'] + [c for c in ['avg_rating', 'num_ratings', 'weighted_score'] if c in pop.columns]
        df = df.merge(pop[cols], on='movieId', how='left')

    result = [movie_to_dict(row) for _, row in df.iterrows()]
    return jsonify({'movies': result, 'method': 'genre'})


@app.route('/api/gems')
def api_gems():
    """Get hidden gems"""
    if DATA['movie_popularity'] is None:
        return jsonify([])

    min_ratings = request.args.get('min_ratings', 100, type=int)
    max_ratings = request.args.get('max_ratings', 1000, type=int)
    min_score = request.args.get('min_score', 4.0, type=float)
    sort_by = request.args.get('sort', 'rating')
    limit = request.args.get('limit', 20, type=int)

    df = DATA['movie_popularity'].copy()

    if 'num_ratings' in df.columns and 'avg_rating' in df.columns:
        df = df[(df['num_ratings'] >= min_ratings) &
                (df['num_ratings'] <= max_ratings) &
                (df['avg_rating'] >= min_score)]

    if 'release_year' not in df.columns:
        df['release_year'] = df['title'].apply(extract_year)

    df = df.sample(frac=1, random_state=np.random.randint(0, 10000))

    if sort_by == 'rating':
        df = df.sort_values('avg_rating', ascending=False)
    elif sort_by == 'rating_asc':
        df = df.sort_values('avg_rating', ascending=True)
    elif sort_by == 'az':
        df = df.sort_values('title', ascending=True)
    elif sort_by == 'za':
        df = df.sort_values('title', ascending=False)
    elif sort_by == 'newest':
        df = df.sort_values('release_year', ascending=False)
    elif sort_by == 'oldest':
        df = df.sort_values('release_year', ascending=True)

    movies = [movie_to_dict(row) for _, row in df.head(limit).iterrows()]
    return jsonify(movies)


@app.route('/api/genre/<genre>')
def api_genre(genre):
    """Get movies by genre"""
    if DATA['movie_popularity'] is None:
        return jsonify([])

    sort_by = request.args.get('sort', 'rating')
    limit = request.args.get('limit', 20, type=int)

    df = DATA['movie_popularity'].copy()
    df = df[df['genres'].str.contains(genre, case=False, na=False)]

    if 'release_year' not in df.columns:
        df['release_year'] = df['title'].apply(extract_year)

    df = df.sample(frac=1, random_state=np.random.randint(0, 10000))

    if sort_by == 'rating':
        if 'weighted_score' in df.columns:
            df = df.sort_values('weighted_score', ascending=False)
        else:
            df = df.sort_values('avg_rating', ascending=False)
    elif sort_by == 'az':
        df = df.sort_values('title', ascending=True)
    elif sort_by == 'za':
        df = df.sort_values('title', ascending=False)
    elif sort_by == 'newest':
        df = df.sort_values('release_year', ascending=False)
    elif sort_by == 'oldest':
        df = df.sort_values('release_year', ascending=True)

    movies = [movie_to_dict(row) for _, row in df.head(limit).iterrows()]
    return jsonify(movies)


@app.route('/api/autocomplete')
def api_autocomplete():
    """Autocomplete search suggestions"""
    query = request.args.get('q', '')
    if not query or len(query) < 2 or DATA['movies'] is None:
        return jsonify([])

    df = DATA['movies'].copy()
    mask = df['title'].str.lower().str.contains(query.lower(), na=False)
    results = df[mask].head(10)

    if DATA['movie_popularity'] is not None:
        pop = DATA['movie_popularity']
        cols = ['movieId'] + [c for c in ['avg_rating', 'num_ratings'] if c in pop.columns]
        results = results.merge(pop[cols], on='movieId', how='left')

    suggestions = []
    for _, row in results.iterrows():
        movie_id = int(row['movieId'])
        title = row['title']
        rating = row.get('avg_rating')

        tmdb_id = None
        imdb_id = None
        if DATA['links'] is not None:
            link = DATA['links'][DATA['links']['movieId'] == movie_id]
            if len(link) > 0:
                if pd.notna(link.iloc[0].get('tmdbId')):
                    tmdb_id = int(link.iloc[0]['tmdbId'])
                if pd.notna(link.iloc[0].get('imdbId')):
                    raw_iid = str(link.iloc[0]['imdbId']).split('.')[0]
                    imdb_id = f"tt{int(raw_iid):07d}"

        suggestions.append({
            'movieId': movie_id,
            'title': title,
            'cleanTitle': str(title).split('(')[0].strip(),
            'year': extract_year(title),
            'rating': round(rating, 1) if pd.notna(rating) else None,
            'tmdbId': tmdb_id,
            'imdbId': imdb_id
        })

    return jsonify(suggestions)


@app.route('/api/featured')
def api_featured():
    """Get a featured movie for hero banner"""
    if DATA['movie_popularity'] is None:
        return jsonify({'error': 'No data'}), 404

    top = DATA['movie_popularity'].head(50).sample(1).iloc[0]
    movie_id = int(top['movieId'])

    tmdb_id = None
    imdb_id = None
    if DATA['links'] is not None:
        link = DATA['links'][DATA['links']['movieId'] == movie_id]
        if len(link) > 0:
            if pd.notna(link.iloc[0].get('tmdbId')):
                tmdb_id = int(link.iloc[0]['tmdbId'])
            if pd.notna(link.iloc[0].get('imdbId')):
                raw_iid = str(link.iloc[0]['imdbId']).split('.')[0]
                imdb_id = f"tt{int(raw_iid):07d}"

    # Get predictions from all 3 models
    predictions = predict_all_models(movie_id)

    return jsonify({
        'movieId': movie_id,
        'title': top['title'],
        'cleanTitle': str(top['title']).split('(')[0].strip(),
        'year': extract_year(top['title']),
        'genres': str(top['genres']).split('|'),
        'avgRating': round(float(top['avg_rating']), 2) if pd.notna(top.get('avg_rating')) else None,
        'numRatings': int(top['num_ratings']) if pd.notna(top.get('num_ratings')) else None,
        'predictions': predictions,
        'tmdbId': tmdb_id,
        'imdbId': imdb_id
    })


@app.route('/api/genre-rows')
def api_genre_rows():
    """Get Netflix-style rows by genre"""
    if DATA['movie_popularity'] is None:
        return jsonify([])

    genres_to_show = ['Action', 'Comedy', 'Drama', 'Thriller', 'Sci-Fi', 'Romance', 'Horror', 'Animation']
    rows = []

    for genre in genres_to_show:
        df = DATA['movie_popularity'].copy()
        df = df[df['genres'].str.contains(genre, case=False, na=False)]
        df = df.sample(frac=1).head(10)

        if len(df) > 0:
            movies = [movie_to_dict(row) for _, row in df.iterrows()]
            rows.append({
                'genre': genre,
                'title': f'Top {genre} Movies',
                'movies': movies
            })

    return jsonify(rows)


@app.route('/api/compare')
def api_compare():
    """Compare two movies"""
    movie1_id = request.args.get('movie1', type=int)
    movie2_id = request.args.get('movie2', type=int)

    if not movie1_id or not movie2_id:
        return jsonify({'error': 'Need two movie IDs'}), 400

    if DATA['movies'] is None:
        return jsonify({'error': 'No data'}), 404

    movies = DATA['movies']

    m1 = movies[movies['movieId'] == movie1_id]
    m2 = movies[movies['movieId'] == movie2_id]

    if len(m1) == 0 or len(m2) == 0:
        return jsonify({'error': 'Movie not found'}), 404

    m1 = m1.iloc[0]
    m2 = m2.iloc[0]

    genres1 = set(str(m1['genres']).split('|'))
    genres2 = set(str(m2['genres']).split('|'))
    common_genres = list(genres1 & genres2)
    unique_to_1 = list(genres1 - genres2)
    unique_to_2 = list(genres2 - genres1)

    genre_sim = len(common_genres) / len(genres1 | genres2) if len(genres1 | genres2) > 0 else 0

    feature_sim = None
    if DATA['genre_encoder'] is not None and DATA['scaler'] is not None:
        f1 = build_features(movie1_id)
        f2 = build_features(movie2_id)
        if f1 is not None and f2 is not None:
            f1_scaled = DATA['scaler'].transform(f1.reshape(1, -1))[0]
            f2_scaled = DATA['scaler'].transform(f2.reshape(1, -1))[0]
            dot = np.dot(f1_scaled, f2_scaled)
            n1, n2 = np.linalg.norm(f1_scaled), np.linalg.norm(f2_scaled)
            if n1 > 0 and n2 > 0:
                feature_sim = round(dot / (n1 * n2) * 100, 1)

    def get_movie_info(movie_id, row):
        info = movie_to_dict(row.to_dict() if hasattr(row, 'to_dict') else row)
        if DATA['movie_popularity'] is not None:
            pop = DATA['movie_popularity'][DATA['movie_popularity']['movieId'] == movie_id]
            if len(pop) > 0:
                info['avgRating'] = round(float(pop.iloc[0]['avg_rating']), 2) if pd.notna(pop.iloc[0].get('avg_rating')) else None
                info['numRatings'] = int(pop.iloc[0]['num_ratings']) if pd.notna(pop.iloc[0].get('num_ratings')) else None
        info['predictions'] = predict_all_models(movie_id)
        return info

    return jsonify({
        'movie1': get_movie_info(movie1_id, m1),
        'movie2': get_movie_info(movie2_id, m2),
        'comparison': {
            'genreSimilarity': round(genre_sim * 100, 1),
            'featureSimilarity': feature_sim,
            'commonGenres': common_genres,
            'uniqueToMovie1': unique_to_1,
            'uniqueToMovie2': unique_to_2
        }
    })


@app.route('/api/random-gem')
def api_random_gem():
    """Get a random hidden gem with full details"""
    if DATA['movie_popularity'] is None:
        return jsonify({'error': 'No data'}), 404

    df = DATA['movie_popularity'].copy()

    if 'num_ratings' in df.columns and 'avg_rating' in df.columns:
        gems = df[(df['num_ratings'] >= 100) &
                  (df['num_ratings'] <= 1000) &
                  (df['avg_rating'] >= 4.0)]
    else:
        gems = df

    if len(gems) == 0:
        return jsonify({'error': 'No gems found'}), 404

    gem = gems.sample(1).iloc[0]
    movie_id = int(gem['movieId'])

    result = movie_to_dict(gem)
    result['predictions'] = predict_all_models(movie_id)

    return jsonify(result)


# =============================================================================
# MAIN
# =============================================================================

# Load models & data ONCE at startup
with app.app_context():
    load_data()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
