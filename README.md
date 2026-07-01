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
| Kaggle historical dataset | Game results + **moneyline** odds | 2008 → **Jan 2023** (cliffs) |
| Kaggle historical dataset | Game results + **point spread** (used as the market baseline where moneylines end) | 2008 → 2026 |

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

### Favorite-Longshot Bias (a calibration pattern, not a bet)

In the de-vigged **moneyline** probabilities, heavy favorites are mildly overpriced: teams the market prices at 75–90% win a few points less often than implied. This is the textbook **favorite-longshot bias**. It shows up as a calibration wobble in the market curve.

The tempting move is to bet it — fade heavy favorites. An earlier version of this project did exactly that and reported **+10.8% ROI**. The significance testing below shows why that number should not be trusted. The bias is a real (small) descriptive pattern; it is **not** a demonstrated edge.

### Layer 2 — Independent Model

A win-probability model was trained on signals that are **independent of the market**: the Elo rating gap, the rolling four-factor differentials (10- and 20-game windows), and schedule/rest features. Two learners — logistic regression and gradient boosting — were averaged into an ensemble.

**Split:** trained on 2013-14 → 2020-21, held out on 2021-22 → 2024-25. Because the point spread reaches every season (see below), the out-of-sample window is **~4,900 games** — all strictly post-training, so nothing the model sees leaks into evaluation.

| Model | Brier | Accuracy | AUC |
|-------|-------|----------|-----|
| Logistic regression | 0.2175 | 65.4% | 0.700 |
| Gradient boosting | 0.2188 | 64.8% | 0.696 |
| **Ensemble** | **0.2177** | **65.3%** | 0.700 |
| Elo baseline | 0.2228 | 64.1% | 0.697 |
| Market (spread-implied) | **0.2057** | **67.9%** | **0.736** |
| Market (de-vigged moneyline)¹ | 0.2143 | 66.4% | 0.708 |

¹ Moneyline row evaluated on the 1,894 test games that still carry moneylines.

The honest takeaway: the model **beats the Elo baseline** and is a **well-calibrated standalone win-probability estimate** — but the **market beats the model**, and the sharper spread-implied line beats it by more. Beating an efficient market head-to-head is the hardest possible target; the real question is whether the model contains information the market does not.

### Significance Testing — the +10.8% does not survive

Point estimates say nothing about whether a result is an edge or a lucky sample. Each strategy's per-bet returns were bootstrapped (10,000 resamples) into a 95% confidence interval on ROI.

| Strategy | Bets | ROI | 95% CI | P(ROI > 0) | Verdict |
|----------|------|-----|--------|-----------|---------|
| Fade heavy home favorites (moneyline 75–90%) | 335 | +10.8% | **[−11.3%, +34.0%]** | 0.17 | not significant |
| Model EV > 0.10 | 945 | −0.8% | [−11.0%, +9.6%] | 0.56 | not significant |
| Always bet favorite (baseline) | 1,894 | −4.0% | [−7.3%, −0.8%] | 0.99 | not significant |

The favorite-longshot rule's confidence interval spans **[−11.3%, +34.0%]** — there is a **17% chance** a true zero-edge strategy scores this well or better on 335 bets. High-variance underdog payouts make 335 bets far too few to distinguish signal from luck. The headline finding was noise.

### The Spread Pivot & Against-the-Spread Backtest

The moneyline data cliffs at January 2023, capping the moneyline backtest at ~1,894 games. But the **point spread is populated for every game through 2026**, and it maps cleanly to win probability — so it becomes the market baseline that reaches the missing seasons. Joining it onto the model dataset matched **100%** of games and roughly **tripled** the out-of-sample universe.

Against-the-spread bets are then settled at the standard **−110** price over the full 2021–25 window (~4,900 games):

| Strategy | Bets | ROI | 95% CI | Verdict |
|----------|------|-----|--------|---------|
| Model edge vs spread (any cutoff) | 2,000–4,900 | −5% to −8% | all below 0 | not significant |
| Fade heavy favorites 6–30 pts | 2,346 | −4.8% | [−8.7%, −1.0%] | not significant |
| Always favorite ATS (baseline) | 4,913 | −4.5% | [−7.2%, −1.9%] | on the vig line |
| Always home ATS (baseline) | 4,913 | −5.5% | [−8.1%, −2.8%] | on the vig line |

The naive baselines landing **exactly on the −4.5% vig line** confirm the settlement is correct. Against that honest benchmark, nothing clears the hold — and the model actually gets *worse* the more selective it becomes, meaning the spots where it most disagrees with the spread are spots where **the spread is right**.

### Closing Line Value — the market encompasses the model

CLV asks a single question: does the model hold information the closing line does not? If not, the close cannot be beaten. Two tests, both on the ~4,900-game out-of-sample set:

**1. Line accuracy** — which implied line better forecasts the actual game margin?

| Forecast | MAE | RMSE |
|----------|-----|------|
| **Closing spread** | **10.35** | **13.30** |
| Model line | 10.83 | 13.83 |

**2. Forecast-encompassing regression** — regress outcomes on both forecasts (as standardized information weights):

| Forecast | Coefficient | 95% CI | Verdict |
|----------|------------|--------|---------|
| **Closing line** | **+1.045** | [+0.92, +1.18] | adds info |
| Model | −0.105 | **[−0.24, +0.02]** | no added info |

The closing line is the sharper forecaster, and once it is known the model's probability adds **zero** predictive information (its coefficient is statistically indistinguishable from zero). The market **encompasses** the model: no CLV is achievable.

### Conclusion

Three independent lines of evidence — significance-tested ROI, line-accuracy comparison, and the encompassing regression — converge on one result:

> **The NBA sides market (moneyline and spread) is efficient with respect to these signals. There is no statistically supported edge, and no closing-line value is achievable.**

This is a deliberately rigorous **negative result**. The model is a competent, well-calibrated win-probability estimator that beats Elo — it simply is not better than an efficient market, which is the expected outcome for a retail-data model. Establishing that *with statistical confidence*, rather than chasing an overfit backtest, is the point.

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
- [x] Expanded odds coverage — point spread joined for 2023-25 (~4,900 OOS games)
- [x] Bootstrap significance testing — 95% CIs on every strategy's ROI
- [x] Against-the-spread backtest at −110 across the full out-of-sample window
- [x] Closing-line-value analysis — forecast-encompassing regression
- [x] **Conclusion: sides market is efficient wrt these signals; no supported edge**
- [ ] Next: hunt in softer markets (totals / over-under, situational spots)

---

## Repo Structure

```
├── fetch_boxscores.py      # Pull team stats from NBA API
├── elo.py                  # Build Elo ratings game-by-game
├── build_features.py       # Four-factor rolling features
├── layer1_calibration.py   # De-vig odds, measure market accuracy
├── join_model_data.py      # Merge all sources into model dataset
├── ingest_spread.py        # Join point-spread market onto the model dataset
├── layer2_model.py         # Train model, 2021-25 OOS, spread market baseline
├── backtest.py             # Moneyline edge detection + flat-stake ROI backtest
├── significance.py         # Bootstrap 95% CIs / p-values on strategy ROI
├── backtest_ats.py         # Against-the-spread backtest at −110 (+ significance)
├── clv_analysis.py         # Line accuracy + forecast-encompassing (CLV) tests
├── fetch_odds.py           # Live + historical odds from The Odds API (unused)
└── data/
    ├── raw/                # Source CSVs (gitignored)
    └── processed/          # Elo output, features, joined data, predictions
```
