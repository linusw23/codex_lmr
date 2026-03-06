from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = BASE_DIR / "Other Files"
DB_DEFAULT_PATH = BASE_DIR / "data" / "lmr.db"

FILE_ACCOUNT = "accountDetails.csv"
FILE_MOVIES = "movieRatingsList.csv"
FILE_PRED = "pred_scores.csv"
FILE_FP_PRED = "fp_pred_scores.csv"

FILM_COLS = [
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
    "NoUserInput",
]


def _database_url() -> str:
    raw = os.getenv("DATABASE_URL")
    if raw:
        if raw.startswith("postgres://"):
            return "postgresql+psycopg2://" + raw[len("postgres://") :]
        return raw
    DB_DEFAULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{DB_DEFAULT_PATH.as_posix()}"


ENGINE = create_engine(_database_url(), future=True)


def _exists(table_name: str) -> bool:
    with ENGINE.begin() as conn:
        if ENGINE.dialect.name == "sqlite":
            row = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name=:t"),
                {"t": table_name},
            ).fetchone()
            return row is not None
        row = conn.execute(
            text(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema='public' AND table_name=:t"
            ),
            {"t": table_name},
        ).fetchone()
        return row is not None


def _empty_table(table_name: str) -> bool:
    with ENGINE.begin() as conn:
        count = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one()
        return count == 0


def database_ready() -> bool:
    if not _exists("accounts") or not _exists("films"):
        return False
    return (not _empty_table("accounts")) and (not _empty_table("films"))


def init_schema() -> None:
    with ENGINE.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS accounts (
                    UserID INTEGER PRIMARY KEY,
                    User TEXT NOT NULL,
                    Password TEXT,
                    Country TEXT,
                    Email TEXT
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS films (
                    tconst TEXT PRIMARY KEY,
                    averageRating REAL,
                    numVotes INTEGER,
                    titleType TEXT,
                    primaryTitle TEXT,
                    startYear INTEGER,
                    runtimeMinutes INTEGER,
                    genre1 TEXT,
                    genre2 TEXT,
                    genre3 TEXT,
                    NoUserInput INTEGER
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS ratings (
                    tconst TEXT NOT NULL,
                    RatingType TEXT NOT NULL,
                    UserID INTEGER NOT NULL,
                    Rating REAL
                )
                """
            )
        )
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_ratings_t ON ratings(tconst)"))
        conn.execute(
            text("CREATE INDEX IF NOT EXISTS idx_ratings_type_user ON ratings(RatingType, UserID)")
        )


def _ensure_account_rows(users: list[str], defaults: dict[str, str] | None = None) -> dict[str, int]:
    defaults = defaults or {"Password": "1234", "Country": "AU", "Email": ""}
    accounts = pd.read_sql("SELECT UserID, User FROM accounts ORDER BY UserID", ENGINE)
    existing = {str(r["User"]): int(r["UserID"]) for _, r in accounts.iterrows()}
    next_id = (max(existing.values()) + 1) if existing else 1

    inserts = []
    for user in users:
        if user not in existing:
            existing[user] = next_id
            inserts.append(
                {
                    "UserID": next_id,
                    "User": user,
                    "Password": defaults["Password"],
                    "Country": defaults["Country"],
                    "Email": defaults["Email"],
                }
            )
            next_id += 1

    if inserts:
        pd.DataFrame(inserts).to_sql("accounts", ENGINE, if_exists="append", index=False)

    return existing


def _user_columns_from_movies(df: pd.DataFrame) -> list[str]:
    if "NoUserInput" not in df.columns:
        return []
    start = df.columns.get_loc("NoUserInput") + 1
    return list(df.columns[start:])


def bootstrap_from_csv(data_dir: Path | None = None) -> None:
    data_dir = data_dir or Path(os.getenv("DATA_DIR", str(DEFAULT_DATA_DIR)))
    if not data_dir.exists():
        return

    if not _empty_table("films") or not _empty_table("accounts"):
        return

    movies_path = data_dir / FILE_MOVIES
    account_path = data_dir / FILE_ACCOUNT
    pred_path = data_dir / FILE_PRED
    fp_path = data_dir / FILE_FP_PRED
    if not movies_path.exists():
        return

    movies = pd.read_csv(movies_path)
    accounts = pd.read_csv(account_path) if account_path.exists() else pd.DataFrame(columns=["User", "Password", "Country", "Email"])
    pred = pd.read_csv(pred_path) if pred_path.exists() else pd.DataFrame(columns=["tconst"])
    fp_pred = pd.read_csv(fp_path) if fp_path.exists() else pd.DataFrame(columns=["tconst"])

    movie_users = _user_columns_from_movies(movies)
    pred_users = [c for c in pred.columns if c != "tconst"]
    fp_users = [c for c in fp_pred.columns if c != "tconst"]
    all_users = list(dict.fromkeys(list(accounts.get("User", [])) + movie_users + pred_users + fp_users))

    # Build accounts with stable UserIDs
    acct_rows = []
    existing_users = set()
    uid = 1
    for _, row in accounts.iterrows():
        user = str(row.get("User", "")).strip()
        if user and user not in existing_users:
            acct_rows.append(
                {
                    "UserID": uid,
                    "User": user,
                    "Password": str(row.get("Password", "")),
                    "Country": str(row.get("Country", "AU")),
                    "Email": str(row.get("Email", "")),
                }
            )
            existing_users.add(user)
            uid += 1
    for user in all_users:
        if user and user not in existing_users:
            acct_rows.append(
                {
                    "UserID": uid,
                    "User": user,
                    "Password": "1234",
                    "Country": "AU",
                    "Email": "",
                }
            )
            existing_users.add(user)
            uid += 1

    with ENGINE.begin() as conn:
        conn.execute(text("DELETE FROM accounts"))
    pd.DataFrame(acct_rows).to_sql("accounts", ENGINE, if_exists="append", index=False)
    user_map = {r["User"]: int(r["UserID"]) for r in acct_rows}

    films = movies[FILM_COLS].copy()
    films["NoUserInput"] = films["NoUserInput"].fillna(True).astype(bool).astype(int)
    with ENGINE.begin() as conn:
        conn.execute(text("DELETE FROM films"))
    pd.DataFrame(films).to_sql("films", ENGINE, if_exists="append", index=False)

    with ENGINE.begin() as conn:
        conn.execute(text("DELETE FROM ratings"))

    def append_by_user(df: pd.DataFrame, rating_type: str):
        user_cols = [c for c in df.columns if c != "tconst"]
        for user in user_cols:
            uid = user_map.get(user)
            if uid is None:
                continue
            tmp = df[["tconst", user]].copy()
            tmp["Rating"] = pd.to_numeric(tmp[user], errors="coerce")
            tmp = tmp.dropna(subset=["Rating"])
            if tmp.empty:
                continue
            out = pd.DataFrame(
                {
                    "tconst": tmp["tconst"],
                    "RatingType": rating_type,
                    "UserID": uid,
                    "Rating": tmp["Rating"],
                }
            )
            out.to_sql("ratings", ENGINE, if_exists="append", index=False)

    if movie_users:
        append_by_user(movies[["tconst"] + movie_users], "User")
    if "tconst" in pred.columns:
        append_by_user(pred, "Pred")
    if "tconst" in fp_pred.columns:
        append_by_user(fp_pred, "FP Pred")


def _accounts_df_for_app() -> pd.DataFrame:
    return pd.read_sql(
        "SELECT User, Password, Country, Email FROM accounts ORDER BY UserID",
        ENGINE,
    )


def _films_df_for_app() -> pd.DataFrame:
    films = pd.read_sql("SELECT * FROM films", ENGINE)
    films["NoUserInput"] = films["NoUserInput"].fillna(1).astype(int).astype(bool)
    user_map = pd.read_sql("SELECT UserID, User FROM accounts ORDER BY UserID", ENGINE)
    ratings = pd.read_sql("SELECT tconst, UserID, Rating FROM ratings WHERE RatingType='User'", ENGINE)
    if ratings.empty:
        return films
    ratings = ratings.merge(user_map, on="UserID", how="left")
    wide = ratings.pivot_table(index="tconst", columns="User", values="Rating", aggfunc="first")
    wide = wide.reset_index()
    out = films.merge(wide, on="tconst", how="left")
    return out


def _pred_df_for_app(rating_type: str) -> pd.DataFrame:
    user_map = pd.read_sql("SELECT UserID, User FROM accounts ORDER BY UserID", ENGINE)
    ratings = pd.read_sql(
        "SELECT tconst, UserID, Rating FROM ratings WHERE RatingType=:rt",
        ENGINE,
        params={"rt": rating_type},
    )
    if ratings.empty:
        out = pd.DataFrame(columns=["tconst"] + list(user_map["User"]))
        return out
    ratings = ratings.merge(user_map, on="UserID", how="left")
    wide = ratings.pivot_table(index="tconst", columns="User", values="Rating", aggfunc="first")
    return wide.reset_index()


def _with_index(df: pd.DataFrame, index_col):
    if index_col is None:
        return df
    if isinstance(index_col, str):
        return df.set_index(index_col)
    return df


def read_table_for_csv(file_name: str, index_col=None):
    if file_name == FILE_ACCOUNT:
        return _with_index(_accounts_df_for_app(), index_col)
    if file_name == FILE_MOVIES:
        return _with_index(_films_df_for_app(), index_col)
    if file_name == FILE_PRED:
        return _with_index(_pred_df_for_app("Pred"), index_col)
    if file_name == FILE_FP_PRED:
        return _with_index(_pred_df_for_app("FP Pred"), index_col)
    raise ValueError(f"Unsupported table mapping: {file_name}")


def _ensure_tconst(df: pd.DataFrame, index_flag: bool) -> pd.DataFrame:
    if "tconst" in df.columns:
        return df.copy()
    if index_flag:
        out = df.reset_index().copy()
        if out.columns[0] != "tconst":
            out = out.rename(columns={out.columns[0]: "tconst"})
        return out
    return df.copy()


def write_table_for_csv(file_name: str, df: pd.DataFrame, index: bool = True) -> None:
    if file_name == FILE_ACCOUNT:
        in_df = df.copy().reset_index(drop=True)
        in_df = in_df[[c for c in ["User", "Password", "Country", "Email"] if c in in_df.columns]]
        existing = pd.read_sql("SELECT UserID, User FROM accounts ORDER BY UserID", ENGINE)
        user_id_map = {str(r["User"]): int(r["UserID"]) for _, r in existing.iterrows()}
        next_id = (max(user_id_map.values()) + 1) if user_id_map else 1

        rows = []
        for _, row in in_df.iterrows():
            user = str(row["User"])
            if user not in user_id_map:
                user_id_map[user] = next_id
                next_id += 1
            rows.append(
                {
                    "UserID": user_id_map[user],
                    "User": user,
                    "Password": str(row.get("Password", "")),
                    "Country": str(row.get("Country", "AU")),
                    "Email": str(row.get("Email", "")),
                }
            )
        with ENGINE.begin() as conn:
            conn.execute(text("DELETE FROM accounts"))
        if rows:
            pd.DataFrame(rows).to_sql("accounts", ENGINE, if_exists="append", index=False)
        return

    if file_name == FILE_MOVIES:
        in_df = _ensure_tconst(df, index_flag=index)
        base = in_df[FILM_COLS].copy()
        base["NoUserInput"] = base["NoUserInput"].fillna(True).astype(bool).astype(int)
        with ENGINE.begin() as conn:
            conn.execute(text("DELETE FROM films"))
        if not base.empty:
            pd.DataFrame(base).to_sql("films", ENGINE, if_exists="append", index=False)

        user_cols = [c for c in in_df.columns if c not in FILM_COLS]
        user_map = _ensure_account_rows(user_cols)
        with ENGINE.begin() as conn:
            conn.execute(text("DELETE FROM ratings WHERE RatingType='User'"))
        if user_cols:
            long_df = in_df[["tconst"] + user_cols].melt(
                id_vars="tconst",
                var_name="User",
                value_name="Rating",
            )
            long_df["Rating"] = pd.to_numeric(long_df["Rating"], errors="coerce")
            long_df = long_df.dropna(subset=["Rating"])
            if not long_df.empty:
                long_df["UserID"] = long_df["User"].map(user_map)
                long_df["RatingType"] = "User"
                long_df[["tconst", "RatingType", "UserID", "Rating"]].to_sql(
                    "ratings", ENGINE, if_exists="append", index=False
                )
        return

    if file_name in (FILE_PRED, FILE_FP_PRED):
        in_df = _ensure_tconst(df, index_flag=index)
        user_cols = [c for c in in_df.columns if c != "tconst"]
        user_map = _ensure_account_rows(user_cols)
        rtype = "Pred" if file_name == FILE_PRED else "FP Pred"
        with ENGINE.begin() as conn:
            conn.execute(text("DELETE FROM ratings WHERE RatingType=:rt"), {"rt": rtype})
        if user_cols:
            long_df = in_df[["tconst"] + user_cols].melt(
                id_vars="tconst",
                var_name="User",
                value_name="Rating",
            )
            long_df["Rating"] = pd.to_numeric(long_df["Rating"], errors="coerce")
            long_df = long_df.dropna(subset=["Rating"])
            if not long_df.empty:
                long_df["UserID"] = long_df["User"].map(user_map)
                long_df["RatingType"] = rtype
                long_df[["tconst", "RatingType", "UserID", "Rating"]].to_sql(
                    "ratings", ENGINE, if_exists="append", index=False
                )
        return

    raise ValueError(f"Unsupported table mapping: {file_name}")


def install_bootstrap(auto_bootstrap: bool = False) -> None:
    init_schema()
    if auto_bootstrap and _empty_table("films"):
        bootstrap_from_csv()
