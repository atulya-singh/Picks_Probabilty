"""
Ingest point-spread data and patch it onto the model dataset.

The moneyline columns in the raw odds file cliff at Jan 2023, but the point
spread is populated for every game through 2026. Since the spread is a market
signal that maps cleanly to win probability, we use it as the market baseline
for the seasons the moneyline can't reach.

This script joins the raw spread onto data/processed/model_data.csv (which
already carries Elo, four-factor features, and results for all seasons) and
writes model_data_spread.csv with one new column:

    home_spread : signed spread from the home team's perspective.
                  negative => home is favored (lays points),
                  positive => home is the underdog (gets points).

Join key is (date, home_team, away_team). The raw file uses lowercase short
codes (gs, no, sa, ...) which are mapped to the model_data abbreviations.
"""

import numpy as np
import pandas as pd

RAW_PATH = "data/raw/nba_2008-2026.csv"
MD_PATH  = "data/processed/model_data.csv"
OUT_PATH = "data/processed/model_data_spread.csv"

# Raw short code -> model_data abbreviation.
TEAM_MAP = {
    "atl": "ATL", "bkn": "BKN", "bos": "BOS", "cha": "CHA", "chi": "CHI",
    "cle": "CLE", "dal": "DAL", "den": "DEN", "det": "DET", "gs":  "GSW",
    "hou": "HOU", "ind": "IND", "lac": "LAC", "lal": "LAL", "mem": "MEM",
    "mia": "MIA", "mil": "MIL", "min": "MIN", "no":  "NOP", "ny":  "NYK",
    "okc": "OKC", "orl": "ORL", "phi": "PHI", "phx": "PHX", "por": "POR",
    "sa":  "SAS", "sac": "SAC", "tor": "TOR", "utah": "UTA", "wsh": "WAS",
}


def load_raw_spread() -> pd.DataFrame:
    raw = pd.read_csv(RAW_PATH)
    raw["home_team"] = raw["home"].map(TEAM_MAP)
    raw["away_team"] = raw["away"].map(TEAM_MAP)

    unmapped = set(raw.loc[raw["home_team"].isna(), "home"]) | \
               set(raw.loc[raw["away_team"].isna(), "away"])
    if unmapped:
        print(f"WARNING: unmapped team codes dropped: {sorted(unmapped)}")

    # Signed home spread: home favored -> home lays points -> negative.
    sp = raw["spread"].abs()
    raw["home_spread"] = np.where(
        raw["whos_favored"] == "home", -sp,
        np.where(raw["whos_favored"] == "away", sp, np.nan),
    )

    raw["date"] = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d")
    keep = ["date", "home_team", "away_team", "home_spread"]
    return raw.dropna(subset=["home_team", "away_team"])[keep]


def main() -> None:
    md  = pd.read_csv(MD_PATH)
    md["date"] = pd.to_datetime(md["date"]).dt.strftime("%Y-%m-%d")
    spread = load_raw_spread()

    merged = md.merge(spread, on=["date", "home_team", "away_team"], how="left")

    # Report match quality overall and for the seasons the moneyline can't reach.
    matched = merged["home_spread"].notna()
    print(f"Spread join: {matched.sum():,}/{len(merged):,} "
          f"({100*matched.mean():.1f}%) model rows matched a spread\n")

    cov = (merged.assign(has=matched)
           .groupby("season_y")["has"].agg(games="size", matched="sum"))
    cov["pct"] = (100 * cov["matched"] / cov["games"]).round(1)
    print("=== spread coverage by season ===")
    print(cov.to_string())

    merged.to_csv(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH}  ({len(merged):,} rows, +home_spread)")


if __name__ == "__main__":
    main()
