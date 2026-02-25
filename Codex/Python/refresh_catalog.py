from pathlib import Path
import os

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Other Files"
MOVIE_FILE = DATA_DIR / "movieRatingsList.csv"


def _load_imdb_catalog(min_votes: int) -> pd.DataFrame:
    basics = pd.read_csv("https://datasets.imdbws.com/title.basics.tsv.gz", sep="\t", na_values=["\\N"])
    ratings = pd.read_csv("https://datasets.imdbws.com/title.ratings.tsv.gz", sep="\t", na_values=["\\N"])

    basics = basics[
        (basics["isAdult"] == 0)
        & (basics["titleType"].isin(["movie", "tvMovie"]))
        & basics["genres"].notna()
        & basics["startYear"].notna()
        & basics["runtimeMinutes"].notna()
    ][["tconst", "titleType", "primaryTitle", "startYear", "runtimeMinutes", "genres"]]

    ratings = ratings[(ratings["numVotes"] >= min_votes)][["tconst", "averageRating", "numVotes"]]

    merged = basics.merge(ratings, on="tconst", how="inner")
    genres = merged["genres"].str.split(",", expand=True)
    merged["genre1"] = genres[0]
    merged["genre2"] = genres[1]
    merged["genre3"] = genres[2]

    out = merged[
        [
            "tconst",
            "averageRating",
            "numVotes",
            "titleType",
            "primaryTitle",
            "startYear",
            "runtimeMinutes",
            "genre1",
            "genre2",
            "genre3",
        ]
    ].copy()
    out["numVotes"] = out["numVotes"].astype(int)
    out["startYear"] = out["startYear"].astype(int)
    out["runtimeMinutes"] = out["runtimeMinutes"].astype(int)
    return out


def main() -> None:
    min_votes = int(os.getenv("IMDB_MIN_VOTES", "1000"))
    existing = pd.read_csv(MOVIE_FILE)

    user_cols = list(existing.columns[10:])
    if "tconst" not in user_cols:
        user_cols.append("tconst")
    ratings_only = existing[user_cols].copy()

    catalog = _load_imdb_catalog(min_votes=min_votes)
    updated = catalog.merge(ratings_only, how="left", on="tconst")
    updated["NoUserInput"] = updated["NoUserInput"].fillna(True)
    updated = updated.sort_values(by="numVotes", ascending=False).reset_index(drop=True)
    updated.to_csv(MOVIE_FILE, index=False)


if __name__ == "__main__":
    main()
