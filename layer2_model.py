"""
Layer 2 — Independent win-probability model.

Trains models on Elo + four-factor rolling form + rest features (signals that
are independent of the betting market), then compares their out-of-sample
probabilities against:
  - the Elo-only baseline (p_elo_home)
  - the de-vigged market baseline (p_true_home, Layer 1)

The whole point of Layer 2 is to produce probabilities that are *good on their
own* but also *diverge* from the market — those divergences are the edges that
backtest.py exploits.

Time-based split (no leakage across seasons):
  train  : 2013-14 .. 2020-21
  test   : 2021-22 .. 2022-23   (held out, never seen in training)

NOTE on the holdout window: the Kaggle moneyline odds only run through the
first ~664 games of 2022-23 (2023-24 and 2024-25 have no odds at all). Because
edge detection and backtesting require market prices, the holdout is set to the
last two seasons that still carry odds, keeping the backtest fully out-of-sample.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score, roc_auc_score

MODEL_DATA  = "data/processed/model_data_spread.csv"
PRED_PATH   = "data/processed/layer2_predictions.csv"
PLOT_PATH   = "data/processed/layer2_calibration.png"

TRAIN_SEASONS = [
    "2013-14", "2014-15", "2015-16", "2016-17",
    "2017-18", "2018-19", "2019-20", "2020-21",
]
# The moneyline dies after 2022-23, but the point spread covers every season, so
# the out-of-sample window now runs through 2024-25. All test seasons are still
# strictly post-training, so nothing the model sees leaks into evaluation.
TEST_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25"]

# Four-factor metrics that exist as home_<m>_roll{10,20} / away_<m>_roll{10,20}
ROLL_METRICS = [
    "efg", "tov_rate", "orb_pct", "ftr", "off_rating",
    "def_efg", "def_tov_rate", "def_ftr", "def_rating",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct symmetric home-minus-away differential features."""
    feats = pd.DataFrame(index=df.index)

    # Team-strength signal: Elo rating gap (already home-court adjusted upstream).
    feats["elo_diff"] = df["elo_home_pre"] - df["elo_away_pre"]

    # Four-factor form differentials at both rolling windows.
    for m in ROLL_METRICS:
        for w in (10, 20):
            feats[f"{m}_diff_roll{w}"] = (
                df[f"home_{m}_roll{w}"] - df[f"away_{m}_roll{w}"]
            )

    # Schedule / fatigue.
    feats["rest_diff"]     = df["home_rest_days"] - df["away_rest_days"]
    feats["home_is_b2b"]   = df["home_is_b2b"].astype(int)
    feats["away_is_b2b"]   = df["away_is_b2b"].astype(int)
    feats["games_played"]  = (df["home_games_played"] + df["away_games_played"]) / 2.0

    return feats


def evaluate(name: str, p: np.ndarray, y: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "model":    name,
        "brier":    brier_score_loss(y, p),
        "log_loss": log_loss(y, p),
        "accuracy": accuracy_score(y, (p >= 0.5).astype(int)),
        "auc":      roc_auc_score(y, p),
    }


def main() -> None:
    df = pd.read_csv(MODEL_DATA)
    df["home_won"] = (df["score_home"] > df["score_away"]).astype(int)

    X = build_features(df)
    y = df["home_won"].values

    train_mask = df["season_y"].isin(TRAIN_SEASONS).values
    test_mask  = df["season_y"].isin(TEST_SEASONS).values

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    print(f"Features ({X.shape[1]}): {list(X.columns)}")
    print(f"Train games: {train_mask.sum():,}   Test games: {test_mask.sum():,}\n")

    # ── Train the two Layer 2 models ────────────────────────────────────────────
    logit = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0),
    )
    logit.fit(X_train, y_train)

    gbm = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.03,
        max_depth=3,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )
    gbm.fit(X_train, y_train)

    p_logit = logit.predict_proba(X_test)[:, 1]
    p_gbm   = gbm.predict_proba(X_test)[:, 1]
    p_ens   = (p_logit + p_gbm) / 2.0   # simple average ensemble

    # ── Baselines on the same test rows ─────────────────────────────────────────
    p_elo    = df.loc[test_mask, "p_elo_home"].values
    p_market = df.loc[test_mask, "p_true_home"].values

    # Spread-implied market probability: a single-feature logistic mapping the
    # signed home spread to P(home win), fit on TRAIN only (no leakage). This is
    # the market baseline that reaches the seasons the moneyline can't. A handful
    # of games have no spread match -> fit on finite rows, leave the rest NaN.
    spread_train = df.loc[train_mask, "home_spread"].values
    spread_test  = df.loc[test_mask,  "home_spread"].values
    fit_ok = np.isfinite(spread_train)
    spread_lr = LogisticRegression().fit(
        spread_train[fit_ok].reshape(-1, 1), y_train[fit_ok]
    )
    p_spread = np.full(len(spread_test), np.nan)
    sp_ok = np.isfinite(spread_test)
    p_spread[sp_ok] = spread_lr.predict_proba(spread_test[sp_ok].reshape(-1, 1))[:, 1]

    results = [
        evaluate("Logistic Regression", p_logit, y_test),
        evaluate("Gradient Boosting",   p_gbm,   y_test),
        evaluate("Ensemble (avg)",      p_ens,   y_test),
        evaluate("Elo baseline",        p_elo,   y_test),
        evaluate("Market (spread)",     p_spread[sp_ok], y_test[sp_ok]),
    ]

    # Market baseline only on rows where a moneyline exists.
    mkt_ok = ~np.isnan(p_market)
    results.append(
        {**evaluate("Market (de-vigged)", p_market[mkt_ok], y_test[mkt_ok]),
         "model": "Market (de-vigged)"}
    )

    res_df = pd.DataFrame(results)
    print("=== Out-of-sample performance (test = 2021-23) ===")
    print(res_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n(Market row evaluated on {mkt_ok.sum():,} of {test_mask.sum():,} test games with moneylines)")

    # ── Save predictions for the backtester ─────────────────────────────────────
    out = df.loc[test_mask, [
        "date", "season_y", "home_team", "away_team",
        "score_home", "score_away",
        "moneyline_home", "moneyline_away", "home_spread", "home_won",
    ]].copy()
    out["p_model"]         = p_ens
    out["p_logit"]         = p_logit
    out["p_gbm"]           = p_gbm
    out["p_elo"]           = p_elo
    out["p_market"]        = p_market      # de-vigged moneyline (2021-23 only)
    out["p_market_spread"] = p_spread      # spread-implied (all test seasons)
    out.to_csv(PRED_PATH, index=False)
    print(f"\nSaved test-set predictions -> {PRED_PATH}  ({len(out):,} rows)")

    # ── Calibration plot: model vs market ───────────────────────────────────────
    plot_calibration(p_ens, p_market, y_test, mkt_ok)


def plot_calibration(p_model, p_market, y, mkt_ok) -> None:
    edges  = np.linspace(0.0, 1.0, 11)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

    for p, mask, color, lab in [
        (p_model, np.ones_like(y, bool), "darkorange", "Layer 2 model (ensemble)"),
        (p_market, mkt_ok,               "steelblue",  "Market (de-vigged)"),
    ]:
        pv, yv = p[mask], y[mask]
        idx = np.digitize(pv, edges) - 1
        idx = np.clip(idx, 0, 9)
        xs, ys = [], []
        for b in range(10):
            sel = idx == b
            if sel.sum() >= 20:
                xs.append(pv[sel].mean())
                ys.append(yv[sel].mean())
        ax.plot(xs, ys, "o-", color=color, label=lab)

    ax.set_xlabel("Predicted probability (home win)")
    ax.set_ylabel("Actual win rate (home)")
    ax.set_title("Layer 2 Calibration — Model vs Market (test 2021-23)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"Saved calibration plot   -> {PLOT_PATH}")


if __name__ == "__main__":
    main()
