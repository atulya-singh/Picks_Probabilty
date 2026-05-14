import time
import pandas as pd
from nba_api.stats.endpoints import LeagueGameLog

OUTPUT_PATH = "data/raw/nba_boxscores_raw.csv"

KEEP_COLS = [
    "season",
    "TEAM_ABBREVIATION",
    "GAME_ID",
    "GAME_DATE",
    "MATCHUP",
    "WL",
    "FGM", "FGA",
    "FG3M",
    "FTM", "FTA",
    "OREB", "DREB",
    "TOV",
    "PTS",
]

# 2013-14 through 2024-25
START_YEAR = 2013
END_YEAR   = 2024
seasons = [f"{y}-{str(y + 1)[-2:]}" for y in range(START_YEAR, END_YEAR + 1)]

all_frames = []
successful  = []

for season in seasons:
    try:
        time.sleep(0.6)
        lg = LeagueGameLog(
            season=season,
            player_or_team_abbreviation="T",
            season_type_all_star="Regular Season",
            timeout=30,
        )
        df = lg.get_data_frames()[0]
        df["season"] = season
        df = df[KEEP_COLS]
        all_frames.append(df)
        successful.append(season)
        print(f"  {season}: {len(df):,} rows")
    except Exception as e:
        print(f"  {season}: FAILED — {e}")

combined = pd.concat(all_frames, ignore_index=True)
combined.to_csv(OUTPUT_PATH, index=False)

print(f"\nTotal rows pulled : {len(combined):,}")
print(f"Seasons retrieved : {', '.join(successful)}")
print(f"Saved to          : {OUTPUT_PATH}")
