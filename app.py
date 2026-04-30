"""
app.py
======
Flask web server for MovieMatch — Smart Movie Recommender.
Owner: Vishnu K
"""

import os

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from recommender import recommender

# Load .env automatically (works locally; ignored on servers that inject env vars)
load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")

app = Flask(__name__)


@app.before_request
def _ensure_model_loaded():
    """Load the model lazily on the first request."""
    if not recommender.is_loaded:
        try:
            recommender.load_data()
        except (FileNotFoundError, ValueError) as exc:
            print(f"Dataset load warning: {exc}")


@app.route("/")
def index():
    """Serve the main application page."""
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def get_recommendations():
    """Return recommendations for a movie title."""
    body = request.get_json(silent=True) or {}
    title = body.get("title", "").strip()

    if not title:
        return jsonify({"error": "Please provide a movie title."}), 400

    if not recommender.is_loaded:
        return jsonify(
            {
                "error": (
                    "Dataset not loaded. Add a supported TMDB CSV file to the project "
                    "folder: movies.csv, tmdb_5000_movies.csv, credits.csv, or "
                    "tmdb_5000_credits.csv."
                )
            }
        ), 503

    result = recommender.recommend(title)
    status = 200 if "error" not in result else 404
    return jsonify(result), status


@app.route("/search")
def search():
    """Autocomplete title suggestions."""
    query = request.args.get("q", "").strip()
    if len(query) < 2:
        return jsonify([])
    return jsonify(recommender.search_titles(query, limit=8))


@app.route("/stats")
def stats():
    """Return simple dataset statistics."""
    return jsonify(recommender.stats())


@app.route("/health")
def health():
    """Return app and dataset readiness info."""
    return jsonify(
        {
            "status": "ok",
            "model_loaded": recommender.is_loaded,
            "movie_count": len(recommender.df) if recommender.is_loaded else 0,
            "data_source": recommender.data_source if recommender.is_loaded else "",
            "data_files": recommender.data_files if recommender.is_loaded else [],
        }
    )


# Lightweight in-memory cache for TMDB metadata
tmdb_cache = {}

@app.route("/metadata")
def get_metadata():
    """Fetch complete movie metadata from TMDB using API key (v3)."""
    title = request.args.get("title", "").strip()
    if not title:
        return jsonify({"error": "No title provided"})

    # Return cached data if available to avoid redundant API calls
    if title in tmdb_cache:
        return jsonify(tmdb_cache[title])

    api_key = TMDB_API_KEY
    if not api_key:
        return jsonify({"error": "TMDB_API_KEY not configured in .env"})

    try:
        # 1. Search for the movie
        search_url = "https://api.themoviedb.org/3/search/movie"
        params = {
            "api_key": api_key,
            "query": title,
            "page": 1,
            "include_adult": "false",
        }
        search_resp = requests.get(search_url, params=params, timeout=5)
        search_resp.raise_for_status()
        results = search_resp.json().get("results", [])
        
        if not results:
            empty_data = {"error": "Not found"}
            tmdb_cache[title] = empty_data
            return jsonify(empty_data)

        first_result = results[0]
        movie_id = first_result.get("id")
        
        # 2. Fetch full details (runtime, text genres, backdrop)
        details_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        details_resp = requests.get(details_url, params={"api_key": api_key}, timeout=5)
        details_resp.raise_for_status()
        details = details_resp.json()

        metadata = {
            "poster_path": details.get("poster_path"),
            "backdrop_path": details.get("backdrop_path"),
            "release_date": details.get("release_date", ""),
            "vote_average": details.get("vote_average", 0),
            "overview": details.get("overview", ""),
            "runtime": details.get("runtime", 0),
            "genres": [g.get("name") for g in details.get("genres", [])]
        }
        
        tmdb_cache[title] = metadata
        return jsonify(metadata)
        
    except Exception as exc:
        return jsonify({"error": str(exc)})


if __name__ == "__main__":
    print("\nStarting MovieMatch — Smart Movie Recommender by Vishnu K")
    print("Open http://127.0.0.1:5000 in your browser\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
