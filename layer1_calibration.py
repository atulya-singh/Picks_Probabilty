import pandas as pd
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

CSV_PATH = "data/raw/nba_2008-2025.csv"
OUTPUT_PATH = "data/processed/layer1_calibration.png"

# ── 1. Load and clean ──────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

df = df[df["regular"] == True]
df = df.dropna(subset=["moneyline_away", "moneyline_home"])
df = df[(df["season"] >= 2014) & (df["season"] <= 2024)]
df = df.reset_index(drop=True)

print(f"Rows after cleaning: {len(df)}")


# ── 2. American odds → raw implied probability ─────────────────────────────────
def american_to_prob(odds: pd.Series) -> pd.Series:
    return np.where(
        odds < 0,
        odds.abs() / (odds.abs() + 100),
        100 / (odds + 100),
    )

df["p_raw_away"] = american_to_prob(df["moneyline_away"])
df["p_raw_home"] = american_to_prob(df["moneyline_home"])


# ── 3. De-vig: Power method ────────────────────────────────────────────────────
def find_k(p_a: float, p_b: float) -> float:
    return brentq(lambda k: p_a**k + p_b**k - 1.0, 0.5, 20.0)

ks = np.array([find_k(a, b) for a, b in zip(df["p_raw_away"], df["p_raw_home"])])
df["p_true_away"] = df["p_raw_away"].values ** ks
df["p_true_home"] = df["p_raw_home"].values ** ks

row_sums = df["p_true_away"] + df["p_true_home"]
assert (row_sums - 1.0).abs().max() < 0.001, "De-vig probabilities don't sum to 1"


# ── 4. Outcome ─────────────────────────────────────────────────────────────────
df["home_won"] = (df["score_home"] > df["score_away"]).astype(int)


# ── 5. Brier Score ─────────────────────────────────────────────────────────────
brier = ((df["p_true_home"] - df["home_won"]) ** 2).mean()
print(f"Brier Score (home team): {brier:.6f}")


# ── 6. Calibration bucketing ───────────────────────────────────────────────────
bucket_edges = np.linspace(0.5, 1.0, 11)
labels = range(10)
df["bucket"] = pd.cut(df["p_true_home"], bins=bucket_edges, labels=labels, include_lowest=True)

cal = (
    df.groupby("bucket", observed=True)
    .agg(
        mean_pred=("p_true_home", "mean"),
        win_rate=("home_won", "mean"),
        count=("home_won", "count"),
    )
    .reset_index()
)

print("\nCalibration table:")
print(f"{'Bucket':<8} {'Mean Pred':>10} {'Win Rate':>10} {'Count':>8}")
for _, row in cal.iterrows():
    print(f"{str(row['bucket']):<8} {row['mean_pred']:>10.4f} {row['win_rate']:>10.4f} {int(row['count']):>8}")


# ── 7. Plot calibration curve ──────────────────────────────────────────────────
x = cal["mean_pred"].values
y = cal["win_rate"].values
n = cal["count"].values

ci = 1.96 * np.sqrt(y * (1 - y) / np.where(n > 0, n, 1))

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot([0.5, 1.0], [0.5, 1.0], "k--", linewidth=1, label="Perfect calibration")
ax.errorbar(x, y, yerr=ci, fmt="none", color="steelblue", alpha=0.6, capsize=3)
sc = ax.scatter(x, y, s=n / n.max() * 300, color="steelblue", zorder=3, label="Buckets (size ∝ count)")

ax.set_xlabel("Mean Predicted Probability (home)")
ax.set_ylabel("Actual Win Rate (home)")
ax.set_title("NBA Moneyline Market Calibration (2014–2024)")
ax.set_xlim(0.48, 1.02)
ax.set_ylim(0.48, 1.02)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150)
print(f"\nPlot saved to {OUTPUT_PATH}")
plt.show()
