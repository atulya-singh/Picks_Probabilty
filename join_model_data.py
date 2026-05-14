import pandas as pd
import numpy as np
from scipy.optimize import brentq

ELO_PATH      = "data/processed/games_with_elo.csv"
FEAT_PATH     = "data/processed/game_features.csv"
OUTPUT_PATH   = "data/processed/model_data.csv"

# Kaggle abbreviation -> standard NBA abbreviation
ELO_TO_NBA = {
    "atl":  "ATL", "bkn":  "BKN", "bos":  "BOS", "cha":  "CHA", "chi":  "CHI",
    "cle":  "CLE", "dal":  "DAL", "den":  "DEN", "det":  "DET", "gs":   "GSW",
    "hou":  "HOU", "ind":  "IND", "lac":  "LAC", "lal":  "LAL", "mem":  "MEM",
    "mia":  "MIA", "mil":  "MIL", "min":  "MIN", "no":   "NOP", "ny":   "NYK",
    "okc":  "OKC", "orl":  "ORL", "phi":  "PHI", "phx":  "PHX", "por":  "POR",
    "sa":   "SAS", "sac":  "SAC", "tor":  "TOR", "utah": "UTA", "wsh":  "WAS",
}

# ── 1. Load ────────────────────────────────────────────────────────────────────
elo  = pd.read_csv(ELO_PATH)
feat = pd.read_csv(FEAT_PATH)

print("=== Team names ===")
print("File 1 (elo) home teams :", sorted(elo["home"].unique()))
print("File 2 (feat) home teams:", sorted(feat["home_team"].unique()))

# ── 2. Recompute de-vigged probabilities (not present in elo CSV) ──────────────
def american_to_prob(odds: pd.Series) -> pd.Series:
    return np.where(
        odds < 0,
        odds.abs() / (odds.abs() + 100),
        100 / (odds + 100),
    )

def find_k(p_a: float, p_b: float) -> float:
    return brentq(lambda k: p_a**k + p_b**k - 1.0, 0.5, 20.0)

ml_mask = elo["moneyline_home"].notna() & elo["moneyline_away"].notna()
elo["p_true_home"] = np.nan
elo["p_true_away"] = np.nan

ml_rows = elo[ml_mask].copy()
p_raw_home = american_to_prob(ml_rows["moneyline_home"])
p_raw_away = american_to_prob(ml_rows["moneyline_away"])
ks = np.array([find_k(a, b) for a, b in zip(p_raw_away, p_raw_home)])
elo.loc[ml_mask, "p_true_home"] = p_raw_home ** ks
elo.loc[ml_mask, "p_true_away"] = p_raw_away ** ks
print(f"\nRecomputed p_true_home/away for {ml_mask.sum():,} rows with moneylines")

# ── 3. Add normalized join keys ────────────────────────────────────────────────
# File 1: date is already YYYY-MM-DD string; normalize team abbr via ELO_TO_NBA
unmapped = set(elo["home"].unique()) - set(ELO_TO_NBA) | set(elo["away"].unique()) - set(ELO_TO_NBA)
if unmapped:
    print(f"\nWARNING: unmapped elo team codes: {sorted(unmapped)}")

elo["_home_abbr"] = elo["home"].map(ELO_TO_NBA)
elo["_away_abbr"] = elo["away"].map(ELO_TO_NBA)
elo["_date_str"]  = pd.to_datetime(elo["date"]).dt.strftime("%Y-%m-%d")
elo["join_key"]   = elo["_date_str"] + "_" + elo["_home_abbr"] + "_" + elo["_away_abbr"]

# File 2: GAME_DATE may already be datetime or string
feat["_date_str"] = pd.to_datetime(feat["GAME_DATE"]).dt.strftime("%Y-%m-%d")
feat["join_key"]  = feat["_date_str"] + "_" + feat["home_team"] + "_" + feat["away_team"]

# ── 4. Inner join ──────────────────────────────────────────────────────────────
n_elo  = len(elo)
n_feat = len(feat)

# Only keep regular season rows in elo that could match feat (2013-14 onward)
elo_rs = elo[elo["regular"] == True].copy()
n_elo_rs = len(elo_rs)

merged = elo_rs.merge(feat, on="join_key", how="inner")
n_matched = len(merged)

# ── 5. Diagnostics ────────────────────────────────────────────────────────────
print(f"\n=== Join diagnostics ===")
print(f"File 1 total rows         : {n_elo:,}")
print(f"File 1 regular season rows: {n_elo_rs:,}")
print(f"File 2 total rows         : {n_feat:,}")
print(f"Matched (inner join)      : {n_matched:,}")
print(f"Unmatched from File 1     : {n_elo_rs - n_matched:,}  ({100*(n_elo_rs - n_matched)/n_elo_rs:.1f}%)")
print(f"Unmatched from File 2     : {n_feat - n_matched:,}  ({100*(n_feat - n_matched)/n_feat:.1f}%)")

unmatched_elo  = elo_rs[~elo_rs["join_key"].isin(merged["join_key"])][["date", "home", "away", "_home_abbr", "_away_abbr", "join_key"]]
unmatched_feat = feat[~feat["join_key"].isin(merged["join_key"])][["GAME_DATE", "home_team", "away_team", "join_key"]]

pct_unmatched = max(
    (n_elo_rs - n_matched) / n_elo_rs,
    (n_feat - n_matched) / n_feat,
)
if pct_unmatched > 0.05:
    print("\nWARNING: >5% unmatched. Showing 10 examples from each side.")
    print("\nUnmatched from File 1 (elo):")
    print(unmatched_elo.head(10).to_string(index=False))
    print("\nUnmatched from File 2 (feat):")
    print(unmatched_feat.head(10).to_string(index=False))
else:
    print(f"\nMatch rate is good (<= 5% unmatched).")

# ── 6. Clean up temp columns and save ─────────────────────────────────────────
drop_cols = ["_home_abbr", "_away_abbr", "_date_str_x", "_date_str_y", "join_key"]
merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])

merged.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved to {OUTPUT_PATH}")
print(f"Shape: {merged.shape}")
print(f"\nColumns ({len(merged.columns)}):")
for col in merged.columns:
    print(f"  {col}")
