from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Other Files"
MOVIE_FILE = DATA_DIR / "movieRatingsList.csv"
PRED_FILE = DATA_DIR / "pred_scores.csv"
FP_PRED_FILE = DATA_DIR / "fp_pred_scores.csv"


def _user_columns(df: pd.DataFrame) -> list[str]:
    return list(df.columns[df.columns.get_loc("NoUserInput") + 1 :])


def _safe_clip(series: pd.Series) -> pd.Series:
    return series.clip(lower=0.0, upper=10.0)


def build_pred_scores(df: pd.DataFrame, users: list[str]) -> pd.DataFrame:
    rated_pool = df[df["NoUserInput"] == False].copy()
    rated_pool["lmr_avg"] = rated_pool[users].mean(axis=1, skipna=True)

    out = pd.DataFrame(index=rated_pool["tconst"])

    for user in users:
        user_known = rated_pool[user].notna()
        if user_known.any():
            user_bias = (rated_pool.loc[user_known, user] - rated_pool.loc[user_known, "lmr_avg"]).mean()
        else:
            user_bias = 0.0

        base = 0.7 * rated_pool["lmr_avg"].fillna(rated_pool["averageRating"])
        imdb_boost = 0.3 * rated_pool["averageRating"]
        pred = _safe_clip(base + imdb_boost + user_bias)

        pred[rated_pool[user].notna()] = np.nan
        out[user] = pred.values

    out.index.name = "tconst"
    return out


def build_fp_pred_scores(df: pd.DataFrame, users: list[str]) -> pd.DataFrame:
    candidates = df[df["NoUserInput"] == True].copy()
    if candidates.empty:
        out = pd.DataFrame(columns=users)
        out.index.name = "tconst"
        return out

    rated_pool = df[df["NoUserInput"] == False].copy()
    rated_pool["lmr_avg"] = rated_pool[users].mean(axis=1, skipna=True)
    global_avg = rated_pool["lmr_avg"].mean()

    out = pd.DataFrame(index=candidates["tconst"])
    for user in users:
        user_known = rated_pool[user].notna()
        if user_known.any():
            user_bias = (rated_pool.loc[user_known, user] - rated_pool.loc[user_known, "lmr_avg"]).mean()
        else:
            user_bias = 0.0

        pred = _safe_clip(
            0.75 * candidates["averageRating"]
            + 0.25 * global_avg
            + user_bias
        )
        out[user] = pred.values

    out.index.name = "tconst"
    return out


def main() -> None:
    df = pd.read_csv(MOVIE_FILE)
    users = _user_columns(df)

    pred = build_pred_scores(df, users)
    fp_pred = build_fp_pred_scores(df, users)

    pred.to_csv(PRED_FILE, index=True)
    fp_pred.to_csv(FP_PRED_FILE, index=True)


if __name__ == "__main__":
    main()
