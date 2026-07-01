# Picks_Probability

I wanted to know if I could beat the NBA betting market with a model built from public data. The short answer turned out to be no. The longer answer is that I spent most of this project making sure I proved that properly, instead of talking myself into a good-looking backtest. This is the writeup.

---

## The idea

Betting markets are efficient but not perfect, so my plan was:

1. Figure out what the market already knows and treat that as the number to beat.
2. Build my own features from historical NBA data that don't depend on the odds.
3. Train a model and look at where it disagrees with the market. If there's money to be made, it's in those disagreements.

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

## Data

| Source | Contents | Seasons |
|--------|----------|---------|
| NBA official API | Team boxscores — FGM, FGA, 3PM, FTA, OREB, DREB, TOV, PTS | 2013–14 → 2024–25 |
| Kaggle historical dataset | Game results + moneyline odds | 2008 → Jan 2023 (then stops) |
| Kaggle historical dataset | Game results + point spread (my market baseline once the moneylines run out) | 2008 → 2026 |

---

## How I built it

### Elo ratings

I gave every team an Elo rating that updates after each game, following FiveThirtyEight's setup:

- Everyone starts at 1505.
- Home court is worth +100 Elo in the matchup.
- Win probability comes off the rating gap through a logistic curve.
- The K-factor (how fast ratings move) scales with margin of victory, so a blowout moves ratings more than a one-point win.
- At the start of each season I regress ratings 25% back toward the mean to account for roster turnover.

On its own, Elo calls about 66% of games correctly (2014–2024). That's a reasonable single-number baseline to build on.

### Four factors

For the actual features I leaned on Dean Oliver's "four factors," which cover most of what separates winning teams from losing ones:

| Factor | Metric | What it measures |
|--------|--------|-----------------|
| Shooting | eFG% | Field goal efficiency, weighting 3-pointers as 1.5x a 2 |
| Ball security | Turnover rate | Turnovers per possession |
| Second chances | Offensive rebound % | Share of own missed shots recovered |
| Free throws | Free throw rate | Free throw attempts relative to field goal attempts |

I compute all four for both offense and defense (how well a team drags down the opponent's four factors) and turn each into a rolling average over the last 10 and 20 games.

The one thing I was strict about was leakage. Every rolling feature is lagged a game (`shift(1)`), so when I predict game N the features only use games 0 through N-1. The model never gets to peek at the game it's scoring. I also throw in days of rest, a back-to-back flag, and how far into the season the game is.

### Layer 1 — reading the market first

Before training anything I looked at the Vegas odds on their own. Moneylines like -220 or +185 convert to implied probabilities, but the two sides always add up to more than 100% — that gap is the vig, the book's margin. I strip it out with the power method, solving for the exponent `k` where:

```
p_away^k + p_home^k = 1.0
```

That gives me a fair probability for each side. I checked those against 11,413 regular season games from 2014–2024.

---

## What I found

### The market is good

The de-vigged market scores a Brier of **0.2063** over those 11,413 games. Vegas is genuinely hard to beat, and that's the number I was trying to top.

### Favorite-longshot bias

The first thing that jumped out was that heavy favorites are slightly overpriced: teams the market puts at 75–90% win a few points less often than that. This is the textbook favorite-longshot bias, and my first instinct was to bet it by fading heavy favorites.

An early version of this project did exactly that and showed **+10.8% ROI**. I was pretty excited about it for about a day. The significance testing further down is why I stopped being excited. The bias is real, but it's a tiny calibration quirk, not something I can actually bet.

### My model (Layer 2)

Then I trained a model on signals that have nothing to do with the odds: the Elo gap, the rolling four-factor differences at both windows, and the rest/schedule stuff. It's a logistic regression and a gradient boosting model averaged together. I trained on 2013-14 → 2020-21 and held out 2021-22 → 2024-25 — everything in the test set comes after the training window, so nothing leaks. Thanks to the spread data (more on that below) the held-out set is about 4,900 games.

| Model | Brier | Accuracy | AUC |
|-------|-------|----------|-----|
| Logistic regression | 0.2175 | 65.4% | 0.700 |
| Gradient boosting | 0.2188 | 64.8% | 0.696 |
| **Ensemble** | **0.2177** | **65.3%** | 0.700 |
| Elo baseline | 0.2228 | 64.1% | 0.697 |
| Market (spread-implied) | **0.2057** | **67.9%** | **0.736** |
| Market (de-vigged moneyline)¹ | 0.2143 | 66.4% | 0.708 |

¹ The moneyline row only covers the 1,894 test games that still have moneylines.

So my model beats Elo and is reasonably well calibrated, but the market still beats it, and the spread-implied line beats it by more. I didn't really expect to win head-to-head against the market. What I actually wanted to know was whether my model knew anything the market didn't.

### Checking whether the +10.8% was real

A single ROI number can't tell you if you got lucky, so I bootstrapped each strategy — resampling its per-bet results 10,000 times to get a confidence interval on the ROI.

| Strategy | Bets | ROI | 95% CI | P(ROI > 0) | Verdict |
|----------|------|-----|--------|-----------|---------|
| Fade heavy home favorites (moneyline 75–90%) | 335 | +10.8% | **[−11.3%, +34.0%]** | 0.17 | not significant |
| Model EV > 0.10 | 945 | −0.8% | [−11.0%, +9.6%] | 0.56 | not significant |
| Always bet favorite (baseline) | 1,894 | −4.0% | [−7.3%, −0.8%] | 0.99 | not significant |

That fade-favorites edge has a 95% interval of **[−11.3%, +34.0%]**. There's a 17% chance you'd see a result that good from a strategy with no edge at all. 335 bets on volatile underdog payouts is just not enough to tell signal from luck, and the headline number was luck.

### Getting more data — the spread pivot

My real problem was sample size, and the reason I only had 1,894 games was that the moneyline data dries up in January 2023. But the point spread is there for every game through 2026, and the spread maps cleanly onto a win probability. So I switched my market baseline over to the spread, which tripled the out-of-sample set to about 4,900 games. The join matched 100% of my games.

With that, I ran an against-the-spread backtest, settling every bet at the standard −110 price:

| Strategy | Bets | ROI | 95% CI | Verdict |
|----------|------|-----|--------|---------|
| Model edge vs spread (any cutoff) | 2,000–4,900 | −5% to −8% | all below 0 | not significant |
| Fade heavy favorites 6–30 pts | 2,346 | −4.8% | [−8.7%, −1.0%] | not significant |
| Always favorite ATS (baseline) | 4,913 | −4.5% | [−7.2%, −1.9%] | on the vig line |
| Always home ATS (baseline) | 4,913 | −5.5% | [−8.1%, −2.8%] | on the vig line |

I ran the naive baselines as a sanity check first: betting every home team or every favorite lands right on the −4.5% vig line, which tells me my settlement math is correct. And against that, nothing beats the vig. The part I found interesting is that my model actually gets *worse* the pickier it gets — the games where it most disagrees with the spread are the ones where the spread is right and I'm wrong.

### Closing line value

This felt like the real test to me. The question is just: does my model know anything the closing line doesn't? If it doesn't, I can't beat the close, and that's the end of it. I checked two ways.

First, which line predicts the actual game margin better:

| Forecast | MAE | RMSE |
|----------|-----|------|
| **Closing spread** | **10.35** | **13.30** |
| Model line | 10.83 | 13.83 |

The closing spread wins. Then I regressed the outcome on both forecasts at once, to see if my model gets any weight once the closing line is in the picture:

| Forecast | Coefficient | 95% CI | Verdict |
|----------|------------|--------|---------|
| **Closing line** | **+1.045** | [+0.92, +1.18] | adds info |
| Model | −0.105 | **[−0.24, +0.02]** | no added info |

It gets basically nothing — the model's coefficient sits at zero (and if anything, slightly negative). The closing line already contains everything my model knows, and then some.

### So did it work?

No, and by this point I'm confident about that. Three separate checks — the bootstrapped ROI, the line-accuracy comparison, and the encompassing regression — all land in the same place: the NBA sides market is efficient for the signals I have. There's no edge I can bet and no closing-line value.

I'm fine calling that a result. My model is a solid win-probability estimator — it beats Elo and it's calibrated — it just isn't smarter than a market that people sharpen for a living. I'd much rather know that for sure than ship a backtest I don't actually trust.

### Where I'd go next

The sides market is the hardest one to beat, so if I keep going I'd look at softer spots instead of trying to out-predict the closing spread — totals (over/under), which I already have the data for, or specific situational angles.

---

## Status

- [x] Data collection — NBA boxscores (2013–2025)
- [x] Elo rating system with season regression
- [x] Four-factor rolling features (10 and 20 game windows)
- [x] Market (Layer 1) calibration and bias analysis
- [x] Full dataset joined for modeling
- [x] Layer 2 model (logistic regression + gradient boosting ensemble)
- [x] Moneyline edge detection and flat-stake backtest
- [x] Expanded coverage — point spread joined for 2023-25 (~4,900 OOS games)
- [x] Bootstrap significance testing — 95% CIs on every strategy
- [x] Against-the-spread backtest at −110 across the full out-of-sample window
- [x] Closing-line-value / forecast-encompassing analysis
- [x] Conclusion: the sides market is efficient for these signals, no bettable edge
- [ ] Next: look at softer markets (totals, situational spots)

---

## Repo structure

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
