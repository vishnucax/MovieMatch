"""
recommender.py
==============
Core ML module for the Smart Movie Recommender.
Implements content-based filtering using TF-IDF vectorization
and cosine similarity.
"""

import ast
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _parse_json_column(val: str) -> str:
    """Extract `name` values from TMDB JSON-like list columns."""
    try:
        items = ast.literal_eval(val)
        return " ".join(item["name"] for item in items if isinstance(item, dict) and "name" in item)
    except Exception:
        return ""


def _parse_name_list(val: str, limit: int | None = None) -> str:
    """Extract names from cast/crew lists, optionally capped to a small subset."""
    try:
        items = ast.literal_eval(val)
        names = [item["name"] for item in items if isinstance(item, dict) and "name" in item]
        if limit is not None:
            names = names[:limit]
        return " ".join(names)
    except Exception:
        return ""


def _parse_director(val: str) -> str:
    """Extract director names from a TMDB crew column."""
    try:
        items = ast.literal_eval(val)
        directors = [
            item["name"]
            for item in items
            if isinstance(item, dict) and item.get("job") == "Director" and "name" in item
        ]
        return " ".join(directors)
    except Exception:
        return ""


class MovieRecommender:
    """Content-based movie recommender with flexible TMDB dataset loading."""

    def __init__(self):
        self.df: pd.DataFrame | None = None
        self.cosine_sim: np.ndarray | None = None
        self.indices: pd.Series | None = None
        self.is_loaded: bool = False
        self.data_source: str = ""
        self.data_files: list[str] = []

    def _resolve_dataset_paths(self, filepath: str | None = None) -> tuple[str | None, str | None]:
        """Find supported dataset files in the current project folder."""
        if filepath:
            return (filepath if os.path.exists(filepath) else None), None

        movie_candidates = ["movies.csv", "tmdb_5000_movies.csv"]
        credit_candidates = ["credits.csv", "tmdb_5000_credits.csv"]

        movies_path = next((name for name in movie_candidates if os.path.exists(name)), None)
        credits_path = next((name for name in credit_candidates if os.path.exists(name)), None)
        return movies_path, credits_path

    def load_data(self, filepath: str | None = None) -> None:
        """Load available movie metadata, then build the similarity model."""
        movies_path, credits_path = self._resolve_dataset_paths(filepath)

        if not movies_path and not credits_path:
            raise FileNotFoundError(
                "Dataset not found in the project folder.\n"
                "Supported files: movies.csv, tmdb_5000_movies.csv, credits.csv, tmdb_5000_credits.csv"
            )

        print("[1/4] Loading dataset...")

        movies_df = None
        credits_df = None

        if movies_path:
            raw_movies = pd.read_csv(movies_path)
            movie_cols = [
                "id",
                "movie_id",
                "title",
                "overview",
                "genres",
                "keywords",
                "vote_average",
                "vote_count",
                "release_date",
            ]
            movies_df = raw_movies[[c for c in movie_cols if c in raw_movies.columns]].copy()

        if credits_path:
            raw_credits = pd.read_csv(credits_path)
            credit_cols = ["movie_id", "title", "cast", "crew"]
            credits_df = raw_credits[[c for c in credit_cols if c in raw_credits.columns]].copy()

        if movies_df is not None and credits_df is not None:
            if "movie_id" not in movies_df.columns and "id" in movies_df.columns:
                movies_df = movies_df.rename(columns={"id": "movie_id"})

            if "movie_id" in movies_df.columns and "movie_id" in credits_df.columns:
                df = movies_df.merge(
                    credits_df.drop(columns=["title"], errors="ignore"),
                    on="movie_id",
                    how="left",
                )
            else:
                df = movies_df.merge(credits_df, on="title", how="left")

            self.data_source = "movies + credits"
            self.data_files = [movies_path, credits_path]
        elif movies_df is not None:
            df = movies_df
            self.data_source = "movies"
            self.data_files = [movies_path]
        else:
            df = credits_df
            self.data_source = "credits"
            self.data_files = [credits_path]

        df.dropna(subset=["title"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        print("[2/4] Preprocessing features...")

        if "genres" in df.columns:
            df["genres_text"] = df["genres"].apply(_parse_json_column)
        else:
            df["genres_text"] = ""

        if "keywords" in df.columns:
            df["keywords_text"] = df["keywords"].apply(_parse_json_column)
        else:
            df["keywords_text"] = ""

        if "cast" in df.columns:
            df["cast_text"] = df["cast"].apply(lambda val: _parse_name_list(val, limit=6))
        else:
            df["cast_text"] = ""

        if "crew" in df.columns:
            df["director_text"] = df["crew"].apply(_parse_director)
            df["crew_text"] = df["crew"].apply(lambda val: _parse_name_list(val, limit=8))
        else:
            df["director_text"] = ""
            df["crew_text"] = ""

        if "overview" in df.columns:
            df["overview"] = df["overview"].fillna("")
        else:
            df["overview"] = ""

        # Use whichever descriptive signals are present so the app still works
        # with credits-only uploads.
        df["combined"] = (
            df["genres_text"].fillna("") + " "
            + df["keywords_text"].fillna("") + " "
            + df["director_text"].fillna("") + " "
            + df["cast_text"].fillna("") + " "
            + df["crew_text"].fillna("") + " "
            + df["overview"].fillna("")
        ).str.strip()

        df = df[df["combined"].str.len() > 0].copy()
        df.reset_index(drop=True, inplace=True)

        if df.empty:
            raise ValueError("Dataset was found, but no usable metadata could be built from it.")

        self.df = df

        print("[3/4] Building TF-IDF matrix...")
        tfidf = TfidfVectorizer(
            stop_words="english",
            max_features=15_000,
            ngram_range=(1, 2),
        )
        tfidf_matrix = tfidf.fit_transform(df["combined"])

        print("[4/4] Computing cosine similarity...")
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        self.indices = pd.Series(df.index, index=df["title"].str.lower()).drop_duplicates()

        self.is_loaded = True
        print(
            f"Model ready: {len(df):,} movies loaded from "
            f"{', '.join(self.data_files)} ({self.data_source}).\n"
        )

    def recommend(self, title: str, n: int = 8) -> dict:
        """Return the top-N most similar movies for the supplied title."""
        if not self.is_loaded:
            return {"error": "Model not loaded. Call load_data() first."}

        title_key = title.strip().lower()
        if title_key not in self.indices:
            candidates = [known_title for known_title in self.indices.index if title_key in known_title]
            if not candidates:
                return {"error": f"'{title}' not found. Try another title."}
            title_key = candidates[0]

        idx = self.indices[title_key]
        sim_scores = sorted(
            enumerate(self.cosine_sim[idx]),
            key=lambda item: item[1],
            reverse=True,
        )[1 : n + 1]

        results = []
        for rank, (movie_idx, score) in enumerate(sim_scores, start=1):
            row = self.df.iloc[movie_idx]
            overview = row.get("overview", "")
            if overview and len(overview) > 220:
                overview = overview[:220] + "..."

            results.append(
                {
                    "rank": rank,
                    "title": row["title"],
                    "similarity": round(float(score) * 100, 1),
                    "genres": row.get("genres_text", "").title(),
                    "overview": overview,
                    "rating": (
                        round(float(row["vote_average"]), 1)
                        if "vote_average" in self.df.columns and pd.notna(row.get("vote_average"))
                        else "N/A"
                    ),
                    "year": (
                        str(row.get("release_date", ""))[:4]
                        if pd.notna(row.get("release_date"))
                        else "N/A"
                    ),
                }
            )

        matched = self.df.iloc[self.indices[title_key]]["title"]
        return {"query": matched, "recommendations": results}

    def search_titles(self, query: str, limit: int = 8) -> list[str]:
        """Return up to `limit` matching titles."""
        if not self.is_loaded:
            return []
        query_key = query.strip().lower()
        matches = [title for title in self.indices.index if query_key in title][:limit]
        return [self.df.iloc[self.indices[match]]["title"] for match in matches]

    def stats(self) -> dict:
        """Return simple dataset stats for the frontend/API."""
        if not self.is_loaded:
            return {}

        genres = self.df["genres_text"].dropna().astype(str).str.split().explode()
        genre_sample = ", ".join(genres.value_counts().head(6).index.tolist()) if not genres.empty else ""

        return {
            "total_movies": int(len(self.df)),
            "avg_rating": (
                round(float(self.df["vote_average"].mean()), 2)
                if "vote_average" in self.df.columns
                else "N/A"
            ),
            "data_source": self.data_source,
            "data_files": self.data_files,
            "genres_sample": genre_sample,
        }


recommender = MovieRecommender()
