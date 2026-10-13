
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).parent
DATA_FILE = (
    PROJECT_DIR
    / "Data"
    / "Processed"
    / "unsw_nb15_combined.csv"
)

unique_hosts = set()
pair_counts = {}
total_flows = 0

for chunk in pd.read_csv(
    DATA_FILE,
    usecols=["srcip", "dstip"],
    chunksize=100_000
):
    chunk["srcip"] = chunk["srcip"].astype("string")
    chunk["dstip"] = chunk["dstip"].astype("string")

    unique_hosts.update(chunk["srcip"].dropna().unique())
    unique_hosts.update(chunk["dstip"].dropna().unique())

    counts = chunk.groupby(
        ["srcip", "dstip"],
        dropna=False
    ).size()

    for pair, count in counts.items():
        pair_counts[pair] = pair_counts.get(pair, 0) + int(count)

    total_flows += len(chunk)

print("GRAPH INSPECTION")
print("----------------")
print(f"Total flows: {total_flows:,}")
print(f"Unique hosts: {len(unique_hosts):,}")
print(f"Unique directed host pairs: {len(pair_counts):,}")
print(f"Repeated flows beyond one per pair: "
      f"{total_flows - len(pair_counts):,}")

print("\nTop 10 most frequent directed host pairs:")
for pair, count in sorted(
    pair_counts.items(),
    key=lambda item: item[1],
    reverse=True
)[:10]:
    print(f"{pair[0]} -> {pair[1]}: {count:,} flows")
