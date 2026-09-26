import os
import pandas as pd

GROUND_TRUTH_PATH = "dataset/train/train_ground_truth.tsv"
CANDIDATE_PATH = "output/candidate_pairs.tsv"
SOURCE1_PATH = "processed/train/source1_processed.tsv"
SOURCE2_PATH = "processed/train/source2_processed.tsv"
OUTPUT_PATH = "eda_reports/blocking_eda.tsv"

gt = pd.read_csv(GROUND_TRUTH_PATH, sep="\t", dtype=str)
candidates = pd.read_csv(CANDIDATE_PATH, sep="\t", dtype=str)
source1 = pd.read_csv(SOURCE1_PATH, sep="\t", dtype=str)
source2 = pd.read_csv(SOURCE2_PATH, sep="\t", dtype=str)

print("Ground truth columns:")
print(gt.columns.tolist())

print("\nGround truth shape:", gt.shape)
print("Candidate shape:", candidates.shape)

print("\nGround truth sample:")
print(gt.head())

print("\nCandidate sample:")
print(candidates.head())

print("\nSource1 rows:", len(source1))
print("Source2 rows:", len(source2))

gt = gt.drop_duplicates()
candidates = candidates.drop_duplicates()

gt_pairs = set(zip(gt.iloc[:, 0], gt.iloc[:, 1]))
candidate_pairs = set(zip(candidates["source1_id"], candidates["source2_id"]))

captured = len(gt_pairs & candidate_pairs)
total_gt = len(gt_pairs)
total_candidates = len(candidate_pairs)

recall = captured / total_gt if total_gt else 0

candidate_counts = candidates.groupby("source1_id").size()

zero_candidates = len(
    set(source1["entity_id"]) - set(candidate_counts.index)
)

stats = {
    "source1_rows": len(source1),
    "source2_rows": len(source2),
    "ground_truth_pairs": total_gt,
    "candidate_pairs": total_candidates,
    "true_matches_captured": captured,
    "blocking_recall": recall,
    "avg_candidates_per_s1": candidate_counts.mean(),
    "median_candidates_per_s1": candidate_counts.median(),
    "p95_candidates_per_s1": candidate_counts.quantile(0.95),
    "p99_candidates_per_s1": candidate_counts.quantile(0.99),
    "zero_candidate_s1": zero_candidates
}

print("\nBlocking EDA")
print("-" * 50)

for key, value in stats.items():
    if "recall" in key:
        print(f"{key}: {value:.6f}")
    elif "candidates_per" in key:
        print(f"{key}: {value:,.2f}")
    else:
        print(f"{key}: {value:,}")

os.makedirs("eda_reports", exist_ok=True)

pd.DataFrame([stats]).to_csv(
    OUTPUT_PATH,
    sep="\t",
    index=False
)

print(f"\nSaved: {OUTPUT_PATH}")