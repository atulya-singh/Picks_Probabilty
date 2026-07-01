"""
Against-the-spread (ATS) backtest with significance built in.

The moneyline backtest was capped at ~1,894 games, too few to distinguish the
favorite-longshot edge from noise. The point spread reaches every season, so
this backtest runs on the full 2021-25 out-of-sample window (~4,900 games) and
settles each pick at the standard -110 price.

Cover convention (home_spread is signed, negative => home favored):
    cover_margin = (score_home - score_away) + home_spread
    > 0  home covers      < 0  away covers      == 0  push

Every strategy is reported with a bootstrap 95% CI on ROI and one-sided p-values,
so a good point estimate never gets mistaken for a proven edge. Break-even ATS
ROI at -110 is about -4.5% (you must win 52.4% just to tread water).

Strategies:
  1. Model edge vs spread  — bet the side the model favours relative to the
                             spread-implied probability, at several edge cutoffs.
  2. Fade heavy favorites  — the favorite-longshot idea in spread form: back the
                             underdog to cover when the favorite is laying a lot.
  3. Naive baselines       — always home ATS / always favorite ATS (sanity: both
                             should land near the -4.5% vig line).
"""

import numpy as np
import pandas as pd

from significance import bootstrap_roi, BREAK_EVEN

PRED_PATH  = "data/processed/layer2_predictions.csv"
WIN_PAYOUT = 100 / 110          # profit per $1 on a winning -110 bet


def settle_ats(bet_home: np.ndarray, cover_margin: np.ndarray) -> np.ndarray:
    """Profit per $1 at -110 for each ATS pick (NaN where no bet, 0 on a push)."""
    profit = np.full(len(bet_home), np.nan)
    push = cover_margin == 0

    h = (bet_home == 1)
    profit[h] = np.where(cover_margin[h] > 0, WIN_PAYOUT, -1.0)
    a = (bet_home == 0)
    profit[a] = np.where(cover_margin[a] < 0, WIN_PAYOUT, -1.0)

    profit[push & ~np.isnan(bet_home)] = 0.0
    return profit


def verdict(stats: dict) -> str:
    if not stats:
        return "no bets"
    if stats["ci_lo_pct"] > 0:
        return "EDGE (CI>0)"
    if stats["ci_lo_pct"] > BREAK_EVEN * 100:
        return "beats vig, not profit"
    return "not significant"


def report(rows: list[dict]) -> None:
    out = []
    for label, bet_home, cover_margin in rows:
        stats = bootstrap_roi(settle_ats(bet_home, cover_margin))
        if not stats:
            continue
        out.append({
            "strategy":    label,
            "bets":        stats["bets"],
            "roi_pct":     stats["roi_pct"],
            "ci95":        f"[{stats['ci_lo_pct']:+.1f}, {stats['ci_hi_pct']:+.1f}]",
            "p(ROI>0)":    stats["p_gt_0"],
            "p(ROI>-4.5)": stats["p_gt_be"],
            "verdict":     verdict(stats),
        })
    print(pd.DataFrame(out).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def main() -> None:
    df = pd.read_csv(PRED_PATH).dropna(
        subset=["home_spread", "p_market_spread", "score_home", "score_away"]
    ).reset_index(drop=True)

    print(f"ATS universe: {len(df):,} out-of-sample games with a spread "
          f"({sorted(df['season_y'].unique())})")
    print(f"Settling at -110 (break-even ROI ~ {BREAK_EVEN*100:.1f}%)\n")

    home_spread  = df["home_spread"].values
    cover_margin = (df["score_home"] - df["score_away"]).values + home_spread
    edge         = df["p_model"].values - df["p_market_spread"].values  # >0: model likes home

    # ── 1. Model edge vs spread ─────────────────────────────────────────────────
    # Bet home to cover when the model likes home more than the market by > t;
    # bet away to cover when it likes home less by > t.
    print("=== Model edge vs spread (bet the side the model favours by > t) ===")
    rows = []
    for t in (0.00, 0.03, 0.05, 0.08):
        bet = np.full(len(df), np.nan)
        bet[edge >  t] = 1
        bet[edge < -t] = 0
        rows.append((f"|edge| > {t:.2f}", bet, cover_margin))
    report(rows)

    # ── 2. Favorite-longshot in spread form ─────────────────────────────────────
    # Back the underdog to cover when a team is laying a lot of points.
    print("\n=== Fade heavy favorites: back the dog to cover ===")
    rows = []
    for lo, hi in [(3, 6), (6, 10), (10, 30), (6, 30)]:
        fav_home = (home_spread <= -lo) & (home_spread > -hi)   # home lays lo..hi
        fav_away = (home_spread >=  lo) & (home_spread <  hi)   # away lays lo..hi
        bet = np.full(len(df), np.nan)
        bet[fav_home] = 0   # home favored -> bet away dog
        bet[fav_away] = 1   # away favored -> bet home dog
        rows.append((f"fade fav {lo}-{hi} pts", bet, cover_margin))
    report(rows)

    # ── 3. Naive baselines ──────────────────────────────────────────────────────
    print("\n=== Naive baselines (should sit near the -4.5% vig line) ===")
    rows = [
        ("Always home ATS",     np.ones(len(df)),                 cover_margin),
        ("Always favorite ATS", (home_spread < 0).astype(float),  cover_margin),
    ]
    report(rows)

    print("\nverdict = does the 95% CI lower bound clear 0%? "
          "(a real ATS edge vs. a lucky point estimate)")


if __name__ == "__main__":
    main()
