# Football Market Value Prediction

Course project for **TDDE64 Sports Analytics** at Linköping University.

The goal is to predict Premier League field players' market values from on-field performance statistics, then benchmark those predictions against Transfermarkt's crowd-sourced values and actual transfer fees. We follow the methodology of *"Data-Driven Models for Predicting Field Player Market Value in European Football"* (IEEE doc 11264761).

---

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url>
cd football-market-value-prediction
uv sync
```

Create a `.env` file in the project root:

```
API_FOOTBALL_KEY=your_key_here
```

Get a free key at [api-football.com](https://api-football.com).

---

## Data

Two data sources are required. Neither is committed to the repo.

### API-Football (performance stats)

Fetched automatically and cached to `data/raw/api_football/`. Run once — subsequent runs read from disk.

```bash
uv run python -m src.ingest
```

The free tier allows 100 requests/day. Fetching all 3 seasons (~22 teams × 3 seasons) takes 2–3 days. Already-cached pages are skipped on re-run.

### Transfermarkt (market values)

Download the *"Football Data from Transfermarkt"* dataset from [Kaggle](https://www.kaggle.com/datasets/davidcariboo/player-scores) and place the CSVs in `data/raw/transfermarkt/`:

```
data/raw/transfermarkt/
  players.csv
  player_valuations.csv
  clubs.csv
  transfers.csv
  appearances.csv
```

---

## Pipeline

```
src/ingest.py     — load API-Football (cached) + Transfermarkt CSVs
src/join.py       — match players across sources by name + date of birth
src/features.py   — build Stage 1 / 2 / 3 feature sets
src/models.py     — train Linear, Ridge, Random Forest, XGBoost
```

Run each module from the repo root:

```bash
uv run python -m src.ingest
uv run python -m src.join
uv run python -m src.features
uv run python -m src.models
```

Or run everything interactively in the notebook:

```bash
uv run jupyter notebook notebooks/analysis.ipynb
```

---

## Feature stages

Mirroring the three-stage design from the reference paper:

| Stage | Features | Count |
|---|---|---|
| **Stage 1** — Basic | Age, position, minutes, appearances, goals, assists, cards | ~10 |
| **Stage 2** — Expanded | Stage 1 + per-90 stats, pass accuracy, shot accuracy, dribble %, duels won % | ~22 |
| **Stage 3** — Domain | Stage 2 + age², position percentiles, Δ-stats vs previous season | ~30+ |

---

## Modeling

- **Target**: `log1p(market_value_eur)` — log-transformed to handle heavy right tail
- **Split**: temporal — train on 2022/23 + 2023/24, test on 2024/25 (no data leakage)
- **Models**: Linear Regression, Ridge, Random Forest, XGBoost
- **Position-aware**: separate DEF / MID / FWD models compared against a global model
- **Goalkeepers excluded** — field players only, following the reference paper

---

## Key results (2022/23 season, 5-fold CV)

| Model | R² (Stage 1) | MAE |
|---|---|---|
| Random Forest | 0.67 | €7M |
| XGBoost | 0.67 | €7M |
| Ridge | 0.45 | €8M |

The reference paper reports R² > 0.80 using multi-season data. Results are expected to improve once 2023/24 and 2024/25 seasons are added.

---

## Project structure

```
data/
  raw/
    api_football/      # cached JSON responses (not committed)
    transfermarkt/     # Kaggle CSVs (not committed)
  processed/           # parquet files (not committed)
src/
  api_football_client.py
  ingest.py
  join.py
  features.py
  models.py
notebooks/
  analysis.ipynb
docs/
  project-plan.md
```

---

## Requirements

See `pyproject.toml`. Key dependencies: `pandas`, `scikit-learn`, `xgboost`, `rapidfuzz`, `requests`, `python-dotenv`.

On macOS, XGBoost requires OpenMP: `brew install libomp`.
