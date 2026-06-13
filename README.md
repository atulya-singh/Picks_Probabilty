# Picks_Probability

A multi-layer NBA win probability model built to identify edges against the betting market.

---

## Project Goal

The core idea is simple: the betting market is generally efficient, but not perfect. This project builds a pipeline to:

1. Establish the market (Vegas) as a baseline probability estimate
2. Build independent features from historical NBA data
3. Train a model whose probabilities diverge from the market in meaningful ways — those divergences are potential edges

---

## Pipeline

```
NBA API (2013–2025)   ──▶  Team boxscores (raw stats)
                               │
                               ├──▶  Elo ratings (team strength over time)
                               │
                               └──▶  Rolling four-factor features (recent form)

Historical odds data  ──▶  De-vigged market probabilities (Layer 1 baseline)

                               All three joined into one model dataset
```

---

## Data Sources

| Source | Contents | Seasons |
|--------|----------|---------|
| NBA official API | Team boxscores — FGM, FGA, 3PM, FTA, OREB, DREB, TOV, PTS | 2013–14 → 2024–25 |
| Kaggle historical dataset | Game results + moneyline odds | 2008 → 2025 |

---

## Methodology

### Elo Ratings

Each team is assigned a numerical strength rating that updates after every game. The system follows FiveThirtyEight's approach:

- Starting rating: **1505** for all teams
- Home court advantage: **+100 Elo points** on the rating difference
- Win probability is derived from the rating gap using a logistic curve
- K-factor (how fast ratings move) is scaled by margin of victory — blowing a team out moves ratings more than a one-point win
- At the start of every season, ratings **regress 25% toward the mean** to account for roster turnover

**Elo model accuracy (2014–2024): ~66%** — a reasonable baseline for a single-signal rating system.

### Four Factors Feature Engineering

Dean Oliver's "four factors" framework explains most of what separates winning from losing teams:

| Factor | Metric | What it measures |
|--------|--------|-----------------|
| Shooting | eFG% | Field goal efficiency, weighting 3-pointers as 1.5x a 2 |
| Ball security | Turnover rate | Turnovers per possession |
| Second chances | Offensive rebound % | Share of own missed shots recovered |
| Free throws | Free throw rate | Free throw attempts relative to field goal attempts |

All four factors are computed for both offense and defense (how well you force the opponent's four factors to suffer). Each stat is then turned into a **rolling average over the last 10 and 20 games**.

**No-leakage guarantee:** all rolling features use a one-game lag (`shift(1)`) — when predicting game N, only games 0 through N-1 are used. The model never sees the game it's predicting.

Additional features: days of rest before the game, back-to-back flag, games played into the season.

### Layer 1 — Market Calibration

Before building a model, the Vegas odds were analyzed as a standalone predictor.

American moneylines (e.g. -220, +185) are converted to implied probabilities. But both sides always sum to more than 100% — that's the **vig** (the bookmaker's profit margin). The Power method strips out the vig by finding an exponent `k` such that:

```
p_away^k + p_home^k = 1.0
```

This produces "true" fair probabilities for each side. These were then evaluated against 11,413 regular season games from 2014–2024.

---

## Key Findings

### Market accuracy is strong, but not perfect

| Metric | Value |
|--------|-------|
| Brier Score (de-vigged market vs. actual outcomes) | **0.2063** |
| Games analyzed | 11,413 regular season games (2014–2024) |

A Brier Score of 0.2063 is competitive — Vegas is genuinely good at this. The goal is to beat it.

### Favorite-Longshot Bias

The single most actionable finding so far: **the market systematically overestimates heavy favorites**.

| Probability range | Market behavior |
|------------------|----------------|
| 50–75% (coin-flip to moderate favorite) | Well-calibrated — predicted win rates closely match actual win rates |
| 75–90% (heavy favorite) | **Overestimates by 3–6 percentage points** — teams priced here win less often than implied |

This is the **favorite-longshot bias** — a well-documented inefficiency in sports betting markets. Bettors tend to overvalue dominant teams, so books shade their lines accordingly, and the implied probabilities end up inflated.

In practical terms: if the market says a team has an 82% chance to win, the actual historical win rate for similarly-priced teams is closer to 77–79%. That gap is consistent and systematic — not noise.

### Layer 2 — Independent Model

A win-probability model was trained on signals that are **independent of the market**: the Elo rating gap, the rolling four-factor differentials (10- and 20-game windows), and schedule/rest features. Two learners — logistic regression and gradient boosting — were averaged into an ensemble.

**Split:** trained on 2013-14 → 2020-21, held out on 2021-22 → 2022-23 (the last two seasons that still carry moneyline odds). The model never sees the holdout.

| Model | Brier | Accuracy | AUC |
|-------|-------|----------|-----|
| Logistic regression | 0.2244 | 64.0% | 0.674 |
| Gradient boosting | 0.2250 | 64.0% | 0.672 |
| **Ensemble** | **0.2244** | **64.1%** | 0.674 |
| Elo baseline | 0.2283 | 63.7% | 0.674 |
| Market (de-vigged) | **0.2143** | **66.4%** | **0.708** |

The honest takeaway: the model **beats the Elo baseline** but the **market still beats the model** on aggregate accuracy. That is expected — the market is hard to beat head-to-head. The value isn't in winning every game; it's in finding the *selective spots* where the model and market disagree and the model is right.

### Edge Detection & Backtesting

Bets are placed only where the model implies positive expected value at the real moneyline price. Flat $1 stakes, settled at the actual American odds, on the 1,894 held-out games with odds. Break-even after the bookmaker's hold sits around **−4.5% ROI**.

| Strategy | Bets | Win rate | ROI |
|----------|------|----------|-----|
| Fade heavy home favorites (market 75–90%) | 335 | 22.7% | **+10.8%** |
| Model EV > 0.10 | 945 | 34.2% | −0.8% |
| Always bet home (baseline) | 1,894 | 56.4% | −3.3% |
| Always bet favorite (baseline) | 1,894 | 66.4% | −4.0% |

The standout is the **favorite-longshot rule** — turning the Layer 1 bias finding into a bet (fade heavily-priced home favorites) returns **+10.8% over 335 bets**, clearly above the vig and above the naive baselines. The EV-threshold strategies hover around break-even: competitive with the market but not a reliable edge on their own.

> **Caveat:** the positive-ROI window is a single ~1,900-game holdout and the favorite-longshot rule fires on only 335 bets. It's a promising, theory-consistent signal — not yet a validated betting system. Larger out-of-sample odds data is the next requirement.

---

## Current Status

- [x] Data collection — NBA boxscores (2013–2025)
- [x] Elo rating system with season regression
- [x] Four-factor rolling feature engineering (10 and 20 game windows)
- [x] Market (Layer 1) calibration and bias analysis
- [x] Full dataset joined and ready for modeling
- [x] Layer 2 model training (logistic regression + gradient boosting ensemble)
- [x] Edge detection — model vs market divergence, bucketed by edge size
- [x] Backtesting framework — flat-stake ROI at real moneyline prices
- [ ] Expand odds coverage to 2023-25 for a larger out-of-sample backtest
- [ ] Probability calibration (isotonic / Platt) on the model outputs
- [ ] Bankroll management (Kelly staking) layered on top of edge detection

---

## Repo Structure

```
├── fetch_boxscores.py      # Pull team stats from NBA API
├── elo.py                  # Build Elo ratings game-by-game
├── build_features.py       # Four-factor rolling features
├── layer1_calibration.py   # De-vig odds, measure market accuracy
├── join_model_data.py      # Merge all sources into model dataset
├── layer2_model.py         # Train model, compare to Elo + market baselines
├── backtest.py             # Edge detection + flat-stake ROI backtest
├── fetch_odds.py           # Live + historical odds from The Odds API
└── data/
    ├── raw/                # Source CSVs and JSON odds snapshots
    └── processed/          # Elo output, features, joined data, predictions
```
