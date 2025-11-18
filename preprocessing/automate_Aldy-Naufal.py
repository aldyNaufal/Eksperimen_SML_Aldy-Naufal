import os
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
from dotenv import load_dotenv

# ======================================================
# 1. Path & ENV
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

# Folder RAW (kotor) di root project
RAW_DIR = PROJECT_ROOT / "data"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Folder PREPROCESSED (bersih) di dalam preprocessing
PREP_DIR = PROJECT_ROOT / "preprocessing" / "data_preprocessing"
PREP_DIR.mkdir(parents=True, exist_ok=True)

BASE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
BASE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"

API_KEY = None  # akan diisi di main()


def load_api_key() -> str:
    """
    Load YOUTUBE_API_KEY dari:
    - environment variable (GitHub Actions pakai secrets)
    - atau dari file .env (lokal)
    """
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)

    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise ValueError(
            "YOUTUBE_API_KEY tidak ditemukan. "
            "Set di .env (lokal) atau di GitHub Secrets (CI)."
        )
    return api_key


# ======================================================
# 2. Genre Keywords
# ======================================================

GENRE_KEYWORDS = {
    "MOBA": [
        "mobile legends", "mlbb", "arena of valor", "aov", "league of legends",
        "lol pc", "wild rift", "dota 2", "ml", "dota"
    ],
    "FPS": [
        "valorant", "csgo", "counter strike", "cs2", "call of duty", "codm",
        "apex legends", "overwatch", "valo", "delta force"
    ],
    "Battle Royale": [
        "pubg", "pubg mobile", "free fire", "ff", "fortnite", "warzone"
    ],
    "RPG": [
        "genshin impact", "honkai star rail", "star rail",
        "zenless zone zero", "zzz", "elden ring", "final fantasy",
        "persona", "rpg"
    ],
    "Horror": [
        "outlast", "amnesia", "phasmophobia", "poppy playtime",
        "fnaf", "five nights at freddy", "horror game", "game horror"
    ],
    "Sandbox": [
        "minecraft", "roblox", "terraria", "sandbox"
    ],
    "Sports": [
        "fifa", "ea fc", "pes", "efootball", "nba 2k", "football manager"
    ],
    "Racing": [
        "forza", "gran turismo", "need for speed", "nfs", "f1 23", "f1 24",
        "assetto corsa"
    ],
    "Strategy": [
        "age of empires", "civilization", "clash of clans", "coc",
        "clash royale", "strategy game"
    ],
    "Casual/Party": [
        "stumble guys", "fall guys", "party game", "jackbox"
    ],
    "Simulation": [
        "bus simulator", "truck simulator", "ets2", "euro truck",
        "simulator", "driving simulator", "farming simulator",
        "train simulator", "flight simulator"
    ],
}


# ======================================================
# 3. Helper YouTube API
# ======================================================

def youtube_search(query, max_results=200, published_after=None, published_before=None):
    global API_KEY
    if not API_KEY:
        raise RuntimeError("API_KEY belum diinisialisasi")

    video_ids = []
    next_page_token = None
    fetched = 0

    while fetched < max_results:
        to_fetch = min(50, max_results - fetched)
        params = {
            "key": API_KEY,
            "part": "snippet",
            "type": "video",
            "q": query,
            "maxResults": to_fetch,
            "order": "viewCount",
            "regionCode": "ID",
        }
        if published_after:
            params["publishedAfter"] = published_after
        if published_before:
            params["publishedBefore"] = published_before
        if next_page_token:
            params["pageToken"] = next_page_token

        resp = requests.get(BASE_SEARCH_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

        items = data.get("items", [])
        for item in items:
            video_ids.append(item["id"]["videoId"])

        fetched += len(items)
        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

        time.sleep(0.2)

    return list(dict.fromkeys(video_ids))


def youtube_get_videos_stats(video_ids):
    global API_KEY
    if not API_KEY:
        raise RuntimeError("API_KEY belum diinisialisasi")

    records = []

    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i+50]
        params = {
            "key": API_KEY,
            "part": "snippet,statistics,contentDetails",
            "id": ",".join(chunk),
        }
        resp = requests.get(BASE_VIDEOS_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

        for item in data.get("items", []):
            vid = item["id"]
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})
            content = item.get("contentDetails", {})

            tags = snippet.get("tags", [])
            if isinstance(tags, list):
                tags_joined = "|".join(tags)
            else:
                tags_joined = ""

            record = {
                "video_id": vid,
                "title": snippet.get("title"),
                "description": snippet.get("description"),
                "tags": tags_joined,
                "channel_id": snippet.get("channelId"),
                "channel_title": snippet.get("channelTitle"),
                "published_at": snippet.get("publishedAt"),
                "view_count": int(stats.get("viewCount", 0)),
                "like_count": int(stats.get("likeCount", 0)),
                "comment_count": int(stats.get("commentCount", 0)),
                "duration": content.get("duration"),
            }
            records.append(record)

        time.sleep(0.2)

    return pd.DataFrame(records)


# ======================================================
# 4. Genre & preprocessing
# ======================================================

def detect_genres_from_text(title: str, description: str, tags: str):
    parts = []
    if title:
        parts.append(title)
    if description:
        parts.append(description)
    if tags:
        parts.append(tags)

    if not parts:
        return []

    text = " ".join(parts).lower()
    found_genres = set()

    for genre, keywords in GENRE_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in text:
                found_genres.add(genre)
                break

    return list(found_genres)


def add_primary_genre_column(df_videos: pd.DataFrame) -> pd.DataFrame:
    genres_all = []
    primary = []

    for _, row in df_videos.iterrows():
        g_list = detect_genres_from_text(
            row.get("title", ""),
            row.get("description", ""),
            row.get("tags", ""),
        )
        genres_all.append(g_list)
        primary.append(g_list[0] if g_list else None)

    df_videos = df_videos.copy()
    df_videos["genres_list"] = genres_all
    df_videos["primary_genre"] = primary
    return df_videos


def clean_for_modelling(df_with_genre: pd.DataFrame, min_samples: int = 20) -> pd.DataFrame:
    """
    - Drop primary_genre yang NaN
    - Filter genre yang jumlah sampelnya < min_samples
    - Tambah kolom 'text' = title + description + tags
    """
    print("[CLEAN] Buang baris tanpa genre...")
    df_clf = df_with_genre.dropna(subset=["primary_genre"]).copy()
    print("  Jumlah data sebelum buang NaN:", len(df_with_genre))
    print("  Jumlah data setelah buang NaN:", len(df_clf))

    genre_counts = df_clf["primary_genre"].value_counts()
    valid_genres = genre_counts[genre_counts >= min_samples].index.tolist()
    df_clf = df_clf[df_clf["primary_genre"].isin(valid_genres)].copy()

    print("  Genre yang dipakai:", valid_genres)
    print("  Jumlah data akhir:", len(df_clf))
    print(df_clf["primary_genre"].value_counts())

    # Buat fitur text gabungan
    def combine_text(row):
        parts = []
        for col in ["title", "description", "tags"]:
            val = row.get(col, "")
            if isinstance(val, str):
                parts.append(val)
        return " ".join(parts)

    df_clf["text"] = df_clf.apply(combine_text, axis=1)

    return df_clf


# ======================================================
# 5. Main pipeline
# ======================================================

def run_pipeline():
    # 5.1 Range waktu scraping
    now = datetime.now(timezone.utc)
    published_after = (now - timedelta(days=90)).isoformat()
    published_before = now.isoformat()

    QUERIES = [
        "game indonesia",
        "gaming indonesia",
        "mobile game indonesia",
        "pc game indonesia",
    ]

    all_video_ids = []
    for q in QUERIES:
        print(f"[SCRAPE] Query: {q}")
        vids = youtube_search(
            query=q,
            max_results=200,
            published_after=published_after,
            published_before=published_before,
        )
        print(f"[SCRAPE]   → {len(vids)} video_id")
        all_video_ids.extend(vids)

    all_video_ids = list(dict.fromkeys(all_video_ids))
    print(f"[SCRAPE] Total unique video_id: {len(all_video_ids)}")

    if not all_video_ids:
        print("[WARN] Tidak ada video ditemukan.")
        return

    # 5.2 Ambil detail video
    df_videos = youtube_get_videos_stats(all_video_ids)
    print("[SCRAPE] Shape df_videos:", df_videos.shape)

    # 5.3 Simpan RAW
    raw_path = RAW_DIR / "videos_raw.csv"
    df_videos.to_csv(raw_path, index=False, encoding="utf-8")
    print(f"[SAVE] RAW -> {raw_path}")

    # 5.4 Tambah genres_list & primary_genre
    df_with_genre = add_primary_genre_column(df_videos)
    with_genre_path = RAW_DIR / "videos_with_genre.csv"
    df_with_genre.to_csv(with_genre_path, index=False, encoding="utf-8")
    print(f"[SAVE] RAW (with genre) -> {with_genre_path}")

    # 5.5 Bersihkan untuk modelling & simpan ke folder preprocessing/data_preprocessing
    df_clean = clean_for_modelling(df_with_genre, min_samples=20)
    clean_path = PREP_DIR / "videos_preprocessed.csv"
    df_clean.to_csv(clean_path, index=False, encoding="utf-8")
    print(f"[SAVE] PREPROCESSED -> {clean_path}")

    print("[INFO] Pipeline preprocessing selesai.")


def main():
    global API_KEY
    API_KEY = load_api_key()
    print("[INFO] API key loaded OK")
    print("[INFO] PROJECT_ROOT:", PROJECT_ROOT)
    run_pipeline()


if __name__ == "__main__":
    main()
