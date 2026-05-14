import pandas as pd
import numpy as np

RAW_PATH    = "data/raw/nba_boxscores_raw.csv"
OUTPUT_PATH = "data/processed/game_features.csv"

ROLL_STATS = [
    "efg", "tov_rate", "orb_pct", "ftr", "off_rating",
    "def_efg", "def_tov_rate", "def_ftr", "def_rating",
]
WINDOWS    = [10, 20]
SCHED_COLS = ["rest_days", "is_b2b", "games_played"]

# ── 1. Load and sort ───────────────────────────────────────────────────────────
df = pd.read_csv(RAW_PATH)
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
df = df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)
print(f"Loaded {len(df):,} rows | {df['GAME_DATE'].min().date()} to {df['GAME_DATE'].max().date()}")

# ── 2. Raw offensive four factors ──────────────────────────────────────────────
df["efg"]         = (df["FGM"] + 0.5 * df["FG3M"]) / df["FGA"]
df["tov_rate"]    = df["TOV"] / (df["FGA"] + 0.44 * df["FTA"] + df["TOV"])
df["ftr"]         = df["FTA"] / df["FGA"]
df["possessions"] = df["FGA"] + 0.44 * df["FTA"] - df["OREB"] + df["TOV"]
df["off_rating"]  = 100 * df["PTS"] / df["possessions"]

# ── 3. Self-join to get opponent stats ─────────────────────────────────────────
opp = df[["GAME_ID", "TEAM_ABBREVIATION", "DREB",
          "efg", "tov_rate", "ftr", "off_rating"]].copy()
opp.columns = ["GAME_ID", "opp_team", "opp_DREB",
               "opp_efg", "opp_tov_rate", "opp_ftr", "opp_off_rating"]

df = df.merge(opp, on="GAME_ID", how="left")
df = df[df["TEAM_ABBREVIATION"] != df["opp_team"]].copy()

# Compute orb_pct and defensive four factors
df["orb_pct"]     = df["OREB"] / (df["OREB"] + df["opp_DREB"])
df["def_efg"]     = df["opp_efg"]
df["def_tov_rate"]= df["opp_tov_rate"]
df["def_ftr"]     = df["opp_ftr"]
df["def_rating"]  = df["opp_off_rating"]

df = df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)

# ── 4. Lagged rolling averages — strict no-leakage ────────────────────────────
# shift(1) inside each group ensures row N only uses rows 0..N-1
for stat in ROLL_STATS:
    for window in WINDOWS:
        df[f"{stat}_roll{window}"] = (
            df.groupby("TEAM_ABBREVIATION")[stat]
            .transform(lambda x, w=window: x.shift(1).rolling(w, min_periods=w).mean())
        )

# ── 5. Schedule features ──────────────────────────────────────────────────────
df["rest_days"]   = df.groupby("TEAM_ABBREVIATION")["GAME_DATE"].transform(
    lambda x: x.diff().dt.days
)
df["is_b2b"]      = (df["rest_days"] == 1).astype(int)
# cumcount within (team, season) → count of games played BEFORE this one
df["games_played"]= df.groupby(["TEAM_ABBREVIATION", "season"]).cumcount()

# ── Verification (run before reshape, on team-level df) ───────────────────────
sample_team = sorted(df["TEAM_ABBREVIATION"].unique())[0]
team_rows = df[df["TEAM_ABBREVIATION"] == sample_team].head(25)
print(f"\nVerification — {sample_team} first 25 games:")
print(team_rows[["GAME_DATE", "efg", "efg_roll10"]].to_string(index=False))
nan_count = team_rows["efg_roll10"].isna().sum()
result    = "PASS" if nan_count == 10 else f"FAIL (got {nan_count} NaN)"
print(f"\nFirst-10-are-NaN check: {result}")

# ── 6. Reshape to one row per GAME_ID ─────────────────────────────────────────
roll_cols     = [f"{s}_roll{w}" for s in ROLL_STATS for w in WINDOWS]
all_feat_cols = roll_cols + SCHED_COLS

df["is_home"] = df["MATCHUP"].str.contains(r"vs\.", regex=True)

home = df[df["is_home"]].copy()
home = home.rename(columns={"TEAM_ABBREVIATION": "home_team"})
home = home.rename(columns={c: f"home_{c}" for c in all_feat_cols})
home_keep = ["GAME_ID", "GAME_DATE", "season", "home_team"] + [f"home_{c}" for c in all_feat_cols]
home = home[home_keep]

away = df[~df["is_home"]].copy()
away = away.rename(columns={"TEAM_ABBREVIATION": "away_team"})
away = away.rename(columns={c: f"away_{c}" for c in all_feat_cols})
away_keep = ["GAME_ID", "away_team"] + [f"away_{c}" for c in all_feat_cols]
away = away[away_keep]

game_df = home.merge(away, on="GAME_ID", how="inner")
game_df = game_df.sort_values("GAME_DATE").reset_index(drop=True)

# ── 7. Drop NaN rows ──────────────────────────────────────────────────────────
before  = len(game_df)
game_df = game_df.dropna().reset_index(drop=True)
print(f"\nDropped {before - len(game_df):,} rows (insufficient rolling history) | {len(game_df):,} remain")

# ── 8. Save ───────────────────────────────────────────────────────────────────
game_df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved to {OUTPUT_PATH}")
print(f"Shape: {game_df.shape}")
print(f"\nColumns ({len(game_df.columns)}):")
for col in game_df.columns:
    print(f"  {col}")
