"""
Edge detection + backtesting.

Takes the Layer 2 out-of-sample predictions and looks for spots where the model
disagrees with the market. A "bet" is placed only when the model's probability
implies positive expected value at the offered moneyline price. Stakes are flat
($1) and bets are settled at the real American odds in the dataset.

Two views:
  1. EV-threshold sweep  — how ROI changes as we demand a larger model edge.
  2. Favorite-longshot   — the Layer 1 finding turned into a rule: fade heavy
                           home favorites (market prob 75-90%) and report ROI.

Everything runs on the held-out 2021-22 / 2022-23 games that the model never
trained on and that still carry moneyline odds.
"""

import numpy as np
import pandas as pd

PRED_PATH = "data/processed/layer2_predictions.csv"


def american_to_decimal(ml: np.ndarray) -> np.ndarray:
    """American moneyline -> decimal payout (stake returned + profit)."""
    ml = ml.astype(float)
    return np.where(ml < 0, 1 + 100 / np.abs(ml), 1 + ml / 100)


def settle(bet_home: np.ndarray, won_home: np.ndarray,
           dec_home: np.ndarray, dec_away: np.ndarray) -> np.ndarray:
    """Profit per $1 stake for each placed bet (NaN where no bet)."""
    profit = np.full(len(bet_home), np.nan)
    # Home bets
    h = bet_home == 1
    profit[h] = np.where(won_home[h] == 1, dec_home[h] - 1, -1.0)
    # Away bets
    a = bet_home == 0
    profit[a] = np.where(won_home[a] == 0, dec_away[a] - 1, -1.0)
    return profit


def summarize(label: str, profit: np.ndarray) -> dict:
    placed = profit[~np.isnan(profit)]
    n = len(placed)
    if n == 0:
        return {"strategy": label, "bets": 0, "win_rate": np.nan,
                "profit": 0.0, "roi_pct": np.nan}
    return {
        "strategy": label,
        "bets": n,
        "win_rate": (placed > 0).mean(),
        "profit": placed.sum(),
        "roi_pct": 100 * placed.sum() / n,   # flat $1 stake => staked == n
    }


def main() -> None:
    df = pd.read_csv(PRED_PATH)
    df = df.dropna(subset=["p_market", "moneyline_home", "moneyline_away"]).reset_index(drop=True)
    print(f"Backtest universe: {len(df):,} held-out games with odds "
          f"({sorted(df['season_y'].unique())})\n")

    p_model  = df["p_model"].values
    won_home = df["home_won"].values
    dec_home = american_to_decimal(df["moneyline_home"].values)
    dec_away = american_to_decimal(df["moneyline_away"].values)

    # Expected value per side under the MODEL's probabilities.
    ev_home = p_model * (dec_home - 1) - (1 - p_model)
    ev_away = (1 - p_model) * (dec_away - 1) - p_model

    # ── 1. EV-threshold sweep ───────────────────────────────────────────────────
    # For each game pick the side with higher model EV; bet only if it clears the
    # threshold. Larger threshold = fewer, higher-conviction bets.
    best_is_home = ev_home >= ev_away
    best_ev      = np.where(best_is_home, ev_home, ev_away)

    print("=== EV-threshold sweep (bet the higher-EV side when EV > t) ===")
    rows = []
    for t in [0.00, 0.02, 0.05, 0.08, 0.10, 0.15]:
        take = best_ev > t
        bet_home = np.where(best_is_home, 1, 0).astype(float)
        bet_home[~take] = np.nan
        rows.append({**summarize(f"EV > {t:.2f}", settle(bet_home, won_home, dec_home, dec_away))})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ── 2. Edge buckets (model prob - market prob) ──────────────────────────────
    # Bet the side the model favours relative to the market; group by edge size.
    edge_home = p_model - df["p_market"].values   # >0: model likes home more than market
    print("\n=== ROI by model-vs-market edge on the home side ===")
    bins = [-1, -0.10, -0.05, -0.02, 0.02, 0.05, 0.10, 1]
    labels = ["<-10%", "-10..-5%", "-5..-2%", "-2..2%", "2..5%", "5..10%", ">10%"]
    df["_edge_bucket"] = pd.cut(edge_home, bins=bins, labels=labels)
    erows = []
    for lab in labels:
        sel = (df["_edge_bucket"] == lab).values
        if sel.sum() == 0:
            continue
        # Positive edge -> back home; negative edge -> back away.
        bet_home = np.where(edge_home[sel] > 0, 1, 0).astype(float)
        profit = settle(bet_home, won_home[sel], dec_home[sel], dec_away[sel])
        erows.append({**summarize(lab, profit)})
    print(pd.DataFrame(erows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ── 3. Favorite-longshot rule (the Layer 1 finding, as a bet) ───────────────
    # Market prices home as a heavy favorite (75-90%): fade it, bet the away dog.
    print("\n=== Favorite-longshot: fade heavy home favorites (market 75-90%) ===")
    fav = (df["p_market"].values >= 0.75) & (df["p_market"].values <= 0.90)
    bet_home = np.full(len(df), np.nan)
    bet_home[fav] = 0  # bet away (the underdog)
    print(pd.DataFrame([summarize("Fade home fav 75-90%",
                                  settle(bet_home, won_home, dec_home, dec_away))]
          ).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # Reference: blindly betting every home team and every favorite.
    print("\n=== Naive baselines ===")
    base = []
    bh = np.ones(len(df)); base.append({**summarize("Always bet home", settle(bh, won_home, dec_home, dec_away))})
    fav_side = (df["p_market"].values >= 0.5).astype(float)  # 1 if home favored
    base.append({**summarize("Always bet favorite", settle(fav_side, won_home, dec_home, dec_away))})
    print(pd.DataFrame(base).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\nNote: a break-even bettor scores ROI ~ -4.5% (the vig). "
          "Strategies above that line are beating the hold; below it are not.")


if __name__ == "__main__":
    main()
