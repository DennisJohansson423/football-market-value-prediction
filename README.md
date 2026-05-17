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

By default, only the Premier League is fetched. To expand to all top-5 European leagues (PL, La Liga, Bundesliga, Serie A, Ligue 1), pass `league_ids` explicitly — see `src/ingest.py` and the notebook for examples.

### Transfermarkt (market values)

Download the *"Football Data from Transfermarkt"* dataset from [Kaggle](https://www.kaggle.com/datasets/davidcariboo/player-scores) and place the CSVs in `data/raw/transfermarkt/`:

```
data/raw/transfermarkt/
  players.csv
  player_valuations.csv
  clubs.csv
  transfers.csv
```

---

## Pipeline

```
src/ingest.py     — load API-Football (cached) + Transfermarkt CSVs
src/join.py       — match players across sources by name + date of birth
src/features.py   — build Stage 1–5 feature sets
src/models.py     — train Linear, Ridge, Random Forest, XGBoost, stacked ensemble
src/evaluate.py   — SHAP feature importance + transfer fee benchmarking
src/predict.py    — interactive terminal tool for player lookups
```

Run each module from the repo root:

```bash
uv run python -m src.ingest
uv run python -m src.join
uv run python -m src.features
uv run python -m src.models
```

Look up a player's predicted value from the terminal:

```bash
uv run python -m src.predict
```

Or run everything interactively in the notebook:

```bash
uv run jupyter notebook notebooks/analysis.ipynb
```

---

## Feature stages

Five progressive feature sets, each building on the previous:

| Stage | Features | Count |
|---|---|---|
| **Stage 1** — Basic | Age, position (DEF/MID/FWD), minutes, appearances, goals, assists, cards | ~10 |
| **Stage 2** — Expanded | Stage 1 + per-90 stats, pass accuracy, shot accuracy, dribble %, duels won % | ~22 |
| **Stage 3** — Domain | Stage 2 + age², position-relative percentile ranks, Δ-stats vs previous season | ~29 |
| **Stage 4** — Context | Stage 3 + goal involvement p90, defensive actions p90, minutes per appearance, **team quality** (`log_team_avg_value`), league dummies | ~39 |
| **Stage 5** — Age curve | Stage 4 + position-specific peak age distance (`age_vs_pos_peak`), sell-on proxy (`years_to_30`), young player flag, age × goal involvement interaction, age × minutes interaction | ~44 |

**Stage 4 is the biggest jump**: the team quality feature (`log_team_avg_value` = log of the average squad market value) captures the fact that playing for a top club is itself a strong signal of quality, lifting R² from 0.616 → 0.776.

---

## Modeling

- **Target**: `log1p(market_value_eur)` — log-transformed to handle the heavy right tail
- **Split**: temporal — train on 2022/23 + 2023/24, test on 2024/25 (no data leakage)
- **Models**: Linear Regression, Ridge, Random Forest, XGBoost, stacked ensemble (RF + XGBoost → Ridge meta-learner)
- **Position-aware**: separate DEF / MID / FWD models compared against a global model
- **Tuning**: RandomizedSearchCV (10 iterations, 3-fold CV) for RF and XGBoost
- **Goalkeepers excluded** — field players only, following the reference paper

### Stacked ensemble

A two-level stack: Random Forest and XGBoost are the base models; Ridge is the meta-learner. The meta-learner is trained exclusively on **out-of-fold predictions** (5-fold CV on the training set) to prevent leakage. Both base models are then refit on the full training set before predicting on the test set.

---

## Key results

Temporal split: train on 2022/23 + 2023/24, test on 2024/25 (2,583 player-seasons).

| Model | Stage | R² | MAE (€M) |
|---|---|---|---|
| XGBoost | Stage 5 | 0.778 | 3.6 |
| Stacked (RF+XGB) | Stage 5 | — | — |
| XGBoost | Stage 4 | 0.776 | 3.5 |
| Random Forest | Stage 5 | 0.764 | 3.9 |
| XGBoost | Stage 3 | 0.616 | 5.6 |
| XGBoost | Stage 2 | 0.546 | 5.9 |
| XGBoost | Stage 1 | 0.485 | 6.6 |

The reference paper reports R² > 0.80 using 24,000+ player-seasons across multiple European leagues. Our best result (R²=0.778) approaches this benchmark on PL-only data with 3 seasons, largely due to the team quality feature added in Stage 4.

### Position-aware results (Stage 5, XGBoost)

| Position | R² | MAE (€M) |
|---|---|---|
| MID | 0.789 | 3.8 |
| FWD | 0.770 | 4.1 |
| DEF | 0.738 | 3.3 |

### Transfer fee benchmark

Predictions from the 2024/25 test set benchmarked against actual paid transfer fees in the **2025/26 summer window**. Transfermarkt wins on MAE — consistent with the finding that transfer fees reflect factors beyond on-field stats (contract length, sell-on clauses, negotiation, hype). The model is better interpreted as a performance-based floor value.

---

## Known limitations

- **Elite youth prospects**: stats-based models systematically undervalue young players with minimal minutes. A highly-rated academy player born in 2005 playing 250 minutes in Ligue 1 looks like a squad filler to the model, even if scouts value them at €20M+. The `is_young` and `years_to_30` features partially address this but cannot compensate for near-zero playing time.
- **Dataset size**: PL-only, 3 seasons ≈ 8,400 player-seasons. The reference paper uses 24,000+. Multi-league support is implemented; additional API quota is needed to fetch it.
- **Transfer fee noise**: fees reflect negotiation and market conditions, not just player quality — so both the model and TM have high MAE on this benchmark.

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
  evaluate.py
  predict.py
notebooks/
  analysis.ipynb
```

---

## Requirements

See `pyproject.toml`. Key dependencies: `pandas`, `scikit-learn`, `xgboost`, `shap`, `rapidfuzz`, `requests`, `python-dotenv`.

On macOS, XGBoost requires OpenMP: `brew install libomp`.
