"""
Phase 1 — Statistical significance of the backtest edges.

backtest.py reports point-estimate ROIs (e.g. the favorite-longshot fade at
+10.8% over 335 bets). A point estimate says nothing about whether that number
is a real edge or the tail of a lucky sample. This script answers that.

For each strategy we reconstruct its per-bet profit vector (settled at the real
American odds, exactly as backtest.py does) and bootstrap it:

  - Resample the bet-level returns with replacement B times.
  - Each resample gives one ROI -> an empirical sampling distribution of ROI.
  - Report the 95% confidence interval (2.5th / 97.5th percentiles).
  - Bootstrap p-values for two one-sided nulls:
        H0a: ROI <= 0%      (is the strategy actually profitable?)
        H0b: ROI <= -4.5%   (does it beat a no-edge bettor paying the vig?)

A strategy whose 95% CI lower bound sits above 0% is a genuine, statistically
supported edge on this sample. One whose CI straddles zero is a hypothesis, not
a finding — no matter how good the point estimate looks.
"""

import numpy as np
import pandas as pd

from backtest import american_to_decimal, settle

PRED_PATH   = "data/processed/layer2_predictions.csv"
N_BOOT      = 10_000
BREAK_EVEN  = -0.045          # approx ROI of a no-edge bettor paying the hold
RNG         = np.random.default_rng(42)


def bootstrap_roi(profit_per_bet: np.ndarray, n_boot: int = N_BOOT) -> dict:
    """Bootstrap the ROI (mean profit per $1 stake) of one strategy.

    Returns the point estimate, a 95% CI, and one-sided p-values against the
    0% and break-even nulls.
    """
    x = profit_per_bet[~np.isnan(profit_per_bet)]
    n = len(x)
    if n == 0:
        return {}

    # Resample bet indices with replacement: shape (n_boot, n).
    idx  = RNG.integers(0, n, size=(n_boot, n))
    rois = x[idx].mean(axis=1)          # one ROI per bootstrap sample

    return {
        "bets":       n,
        "roi_pct":    100 * x.mean(),
        "ci_lo_pct":  100 * np.percentile(rois, 2.5),
        "ci_hi_pct":  100 * np.percentile(rois, 97.5),
        # p = fraction of resamples that fail to clear the null threshold.
        "p_gt_0":     (rois <= 0.0).mean(),
        "p_gt_be":    (rois <= BREAK_EVEN).mean(),
    }


def strategy_profits(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Rebuild each strategy's per-bet profit vector from the predictions."""
    p_model  = df["p_model"].values
    p_market = df["p_market"].values
    won_home = df["home_won"].values
    dec_home = american_to_decimal(df["moneyline_home"].values)
    dec_away = american_to_decimal(df["moneyline_away"].values)

    ev_home = p_model * (dec_home - 1) - (1 - p_model)
    ev_away = (1 - p_model) * (dec_away - 1) - p_model
    best_is_home = ev_home >= ev_away
    best_ev      = np.where(best_is_home, ev_home, ev_away)

    out: dict[str, np.ndarray] = {}

    # Headline finding: fade heavy home favorites (market 75-90%) -> bet the dog.
    fav = (p_market >= 0.75) & (p_market <= 0.90)
    bet = np.full(len(df), np.nan); bet[fav] = 0
    out["Fade home fav 75-90%"] = settle(bet, won_home, dec_home, dec_away)

    # EV-threshold strategies.
    for t in (0.10, 0.05):
        take = best_ev > t
        bet  = np.where(best_is_home, 1, 0).astype(float)
        bet[~take] = np.nan
        out[f"Model EV > {t:.2f}"] = settle(bet, won_home, dec_home, dec_away)

    # Naive baselines (should NOT be significant edges).
    out["Always bet home"] = settle(np.ones(len(df)), won_home, dec_home, dec_away)
    fav_side = (p_market >= 0.5).astype(float)
    out["Always bet favorite"] = settle(fav_side, won_home, dec_home, dec_away)

    return out


def main() -> None:
    df = pd.read_csv(PRED_PATH).dropna(
        subset=["p_market", "moneyline_home", "moneyline_away"]
    ).reset_index(drop=True)

    print(f"Universe: {len(df):,} held-out games with odds "
          f"({sorted(df['season_y'].unique())})")
    print(f"Bootstrap: {N_BOOT:,} resamples | break-even ROI = {BREAK_EVEN*100:.1f}%\n")

    rows = []
    for label, profit in strategy_profits(df).items():
        stats = bootstrap_roi(profit)
        rows.append({
            "strategy":    label,
            "bets":        stats["bets"],
            "roi_pct":     stats["roi_pct"],
            "ci95":        f"[{stats['ci_lo_pct']:+.1f}, {stats['ci_hi_pct']:+.1f}]",
            "p(ROI>0)":    stats["p_gt_0"],
            "p(ROI>-4.5)": stats["p_gt_be"],
            "verdict":     verdict(stats),
        })

    res = pd.DataFrame(rows)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nverdict = is the 95% CI lower bound above 0%? "
          "(a real, sample-supported edge vs. just a good point estimate)")


def verdict(stats: dict) -> str:
    if stats["ci_lo_pct"] > 0:
        return "EDGE (CI>0)"
    if stats["ci_lo_pct"] > BREAK_EVEN * 100:
        return "beats vig, not profit"
    return "not significant"


if __name__ == "__main__":
    main()
