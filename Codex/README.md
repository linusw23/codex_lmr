# Liny's Movie Recommender (Codex Upgrade)

This folder is an upgraded fork of your `Original` project, focused on:

- portability (no machine-specific template paths),
- safer secrets (env vars),
- automated nightly refresh (catalog + predictions),
- deployability outside PythonAnywhere (Docker/Gunicorn),
- quick global UI refresh via shared stylesheet.
- database-backed storage for accounts/films/ratings.

## What Changed

1. `Python/flask_app.py`
- Removed hardcoded `/home/...` template paths.
- Added portable project-root path handling.
- Added `/assets/<path>` route.
- Secret key now comes from `FLASK_SECRET_KEY`.
- Added database bridge so legacy CSV reads/writes map to DB tables.

2. `Python/filmRecommender.py`
- Removed hardcoded TMDB key usage.
- Reads `TMDB_API_KEY` from env.
- Removed import-time debug print side effect.

3. `Python/film_recommender_update.py`
- Added clean helper module for homepage/stat functions used by Flask app.

4. `assets/style.css`
- New shared stylesheet used by all pages for a cleaner look/feel.

5. Automated refresh pipeline
- `Python/refresh_catalog.py`: pulls IMDb titles/ratings and merges with user ratings.
- `Python/rebuild_predictions.py`: rebuilds `pred_scores.csv` and `fp_pred_scores.csv`.
- `Python/nightly_refresh.py`: orchestrates both steps.
- `scripts/run_nightly_refresh.ps1`: Windows-friendly runner.

6. Hosting setup
- `requirements.txt`
- `Dockerfile`
- `.env.example`
- `.github/workflows/nightly-refresh.yml` (optional GitHub scheduler)

## Run Locally

1. Create venv and install deps:
```bash
pip install -r requirements.txt
```

2. Set environment variables from `.env.example`.

3. Start app:
```bash
cd Python
python flask_app.py
```

4. Run nightly refresh manually:
```bash
python Python/nightly_refresh.py
```

## Hosting Recommendation

Best path from current state: **Render** (or **Railway**) using Docker.

- Why: easy env var management, simple web service deploy, and scheduled jobs.
- Web service command is already in `Dockerfile`.
- Add `TMDB_API_KEY` and `FLASK_SECRET_KEY` in platform secrets.
- Prefer external Postgres by setting `DATABASE_URL`.

## Database Model

App now supports this DB structure:
- `accounts(UserID, User, Password, Country, Email)`
- `films(tconst, averageRating, numVotes, titleType, primaryTitle, startYear, runtimeMinutes, genre1, genre2, genre3, NoUserInput)`
- `ratings(tconst, RatingType, UserID, Rating)` with `RatingType` in `User`, `Pred`, `FP Pred`.

On first run, DB bootstraps from existing CSV files automatically.

## Next Upgrade Steps (Recommended)

1. Move CSV storage to SQLite/Postgres (concurrency + durability).
2. Replace plain-text password storage with salted hashing (`werkzeug.security`).
3. Split monolithic Flask file into blueprints/services/repositories.
4. Replace baseline prediction rebuild with your CatBoost pipeline in a separate batch worker.
