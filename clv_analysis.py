"""
Closing-line-value analysis (the answerable version).

This dataset carries only the closing spread, not an opening line, so textbook
open->close CLV (did the line move toward my bet?) can't be computed. But CLV is
really a proxy for one question: does the model contain information the closing
line doesn't? If it doesn't, you cannot beat the close, and CLV is impossible by
construction. That question IS answerable here, two ways:

  1. Line accuracy — the closing spread and the model each imply a predicted game
     margin. Which is the better forecast of the actual margin (lower error)? If
     the close wins, the model has no number worth betting.

  2. Forecast-encompassing regression — regress the outcome on BOTH forecasts
     (closing-line prob and model prob, as standardized logits). If the market
     coefficient is large and the model coefficient is ~0, the close statistically
     "encompasses" the model: it already contains the model's information and
     more. Coefficients get bootstrap CIs (no statsmodels needed).

Everything runs on the 2021-25 out-of-sample predictions.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

PRED_PATH = "data/processed/layer2_predictions.csv"
N_BOOT    = 4000
RNG       = np.random.default_rng(42)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main() -> None:
    df = pd.read_csv(PRED_PATH).dropna(
        subset=["home_spread", "p_market_spread", "p_model", "score_home", "score_away"]
    ).reset_index(drop=True)
    print(f"CLV universe: {len(df):,} out-of-sample games "
          f"({sorted(df['season_y'].unique())})\n")

    home_spread   = df["home_spread"].values
    p_market      = df["p_market_spread"].values
    p_model       = df["p_model"].values
    actual_margin = (df["score_home"] - df["score_away"]).values
    y             = df["home_won"].values

    # ── 1. Line accuracy: which implied margin forecasts the real margin? ────────
    # Recover the exact spread->prob logistic (a, b) from the saved pairs, then
    # invert it to turn the model's win prob into a model-implied spread.
    b, a = np.polyfit(home_spread, logit(p_market), 1)   # logit(p) = b*spread + a
    model_line = (logit(p_model) - a) / b                # model's implied spread

    # A line's predicted home margin is minus the (home) spread.
    market_pred = -home_spread
    model_pred  = -model_line

    print("=== Line accuracy: forecasting the actual game margin ===")
    acc = pd.DataFrame([
        {"forecast": "Closing spread", "MAE": np.abs(actual_margin - market_pred).mean(),
         "RMSE": np.sqrt(((actual_margin - market_pred) ** 2).mean())},
        {"forecast": "Model line",     "MAE": np.abs(actual_margin - model_pred).mean(),
         "RMSE": np.sqrt(((actual_margin - model_pred) ** 2).mean())},
    ])
    print(acc.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("(lower is better; the sharper forecaster is the one you'd need to beat)\n")

    # ── 2. Forecast-encompassing regression ─────────────────────────────────────
    # Standardize both logit forecasts so their coefficients are directly
    # comparable as "information weights", then regress the outcome on both.
    f_mkt = standardize(logit(p_market))
    f_mdl = standardize(logit(p_model))
    X = np.column_stack([f_mkt, f_mdl])

    coef = fit_coef(X, y)
    boot = np.array([fit_coef(X[idx], y[idx])
                     for idx in RNG.integers(0, len(y), size=(N_BOOT, len(y)))])

    print("=== Forecast-encompassing regression: home_won ~ close + model ===")
    rows = []
    for name, c, col in [("Closing line", coef[0], boot[:, 0]),
                         ("Model",        coef[1], boot[:, 1])]:
        lo, hi = np.percentile(col, [2.5, 97.5])
        p_gt0  = (col <= 0).mean()
        rows.append({
            "forecast":   name,
            "coef":       c,
            "ci95":       f"[{lo:+.3f}, {hi:+.3f}]",
            "p(coef>0)":  p_gt0,
            "verdict":    "adds info" if lo > 0 else "no added info",
        })
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nReading it: a large, significant closing-line coefficient with a model\n"
          "coefficient whose CI straddles 0 means the market encompasses the model —\n"
          "the close already holds the model's information, so no CLV is available.")


def standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()


def fit_coef(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Coefficients of a logistic fit; C large => negligible regularization."""
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(X, y)
    return lr.coef_[0]


if __name__ == "__main__":
    main()
