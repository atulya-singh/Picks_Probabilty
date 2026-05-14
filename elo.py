import pandas as pd
import numpy as np

CSV_PATH = "data/raw/nba_2008-2025.csv"
OUTPUT_PATH = "data/processed/games_with_elo.csv"
INITIAL_ELO = 1505.0
HOME_ADV = 100
REGRESS_W = 0.75

# ── 1. Load and prepare ────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
df = df.dropna(subset=["score_home", "score_away"])

print(f"Total rows loaded: {len(df)}")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

# ── 2. Initialize Elo ratings ──────────────────────────────────────────────────
elo = {}  # team -> current rating

# ── 3-7. Process games in chronological order ──────────────────────────────────
elo_home_pre_list  = []
elo_away_pre_list  = []
p_elo_home_list    = []
elo_home_post_list = []
elo_away_post_list = []

current_season = None

for row in df.itertuples(index=False):
    # Season-to-season regression (before first game of each new season)
    if current_season is None:
        current_season = row.season
    elif row.season != current_season:
        for team in elo:
            elo[team] = REGRESS_W * elo[team] + (1 - REGRESS_W) * INITIAL_ELO
        current_season = row.season

    # First-time team init
    if row.home not in elo:
        elo[row.home] = INITIAL_ELO
    if row.away not in elo:
        elo[row.away] = INITIAL_ELO

    elo_h = elo[row.home]
    elo_a = elo[row.away]

    # Win probability (home court +100 in diff only)
    elo_diff = elo_h + HOME_ADV - elo_a
    p_home = 1.0 / (10.0 ** (-elo_diff / 400.0) + 1.0)

    # Outcome
    home_won = int(row.score_home > row.score_away)
    W_home   = home_won
    W_away   = 1 - W_home

    # K-factor: FiveThirtyEight MOV adjustment
    MOV = abs(row.score_home - row.score_away)
    elo_diff_winner = (elo_h - elo_a) if home_won else (elo_a - elo_h)
    K = 20.0 * ((MOV + 3) ** 0.8) / (7.5 + 0.006 * elo_diff_winner)

    # Update ratings
    elo_h_new = elo_h + K * (W_home - p_home)
    elo_a_new = elo_a + K * (W_away - (1.0 - p_home))

    elo[row.home] = elo_h_new
    elo[row.away] = elo_a_new

    elo_home_pre_list.append(elo_h)
    elo_away_pre_list.append(elo_a)
    p_elo_home_list.append(p_home)
    elo_home_post_list.append(elo_h_new)
    elo_away_post_list.append(elo_a_new)

# ── 8. Add columns and save ────────────────────────────────────────────────────
df["elo_home_pre"]  = elo_home_pre_list
df["elo_away_pre"]  = elo_away_pre_list
df["p_elo_home"]    = p_elo_home_list
df["elo_home_post"] = elo_home_post_list
df["elo_away_post"] = elo_away_post_list

df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved to {OUTPUT_PATH}")

# Final Elo ratings sorted descending
print("\nFinal Elo ratings (all teams, sorted):")
for team, rating in sorted(elo.items(), key=lambda x: x[1], reverse=True):
    print(f"  {team:>5}: {rating:.1f}")

# Accuracy and Brier score for seasons 2014+
eval_df = df[df["season"] >= 2014].copy()
eval_df["home_won"] = (eval_df["score_home"] > eval_df["score_away"]).astype(int)

favored_home = eval_df["p_elo_home"] > 0.5
correct = (favored_home == eval_df["home_won"].astype(bool)).sum()
accuracy = correct / len(eval_df)
print(f"\nAccuracy (2014+): {accuracy:.4f}  ({correct}/{len(eval_df)} games)")

brier = ((eval_df["p_elo_home"] - eval_df["home_won"]) ** 2).mean()
print(f"Brier Score (2014+): {brier:.6f}")
