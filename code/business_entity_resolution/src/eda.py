from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# IMPORT PREPROCESSING
# ============================================================

SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing import preprocess_dataframe


# ============================================================
# PATHS
# ============================================================

def find_project_root() -> Path:
    current_file = Path(__file__).resolve()

    for parent in [current_file.parent, *current_file.parents]:
        if (
            (parent / "dataset" / "train").is_dir()
            and (parent / "dataset" / "test").is_dir()
        ):
            return parent

    raise FileNotFoundError(
        "Could not find project root containing dataset/train and dataset/test."
    )


ROOT_DIR = find_project_root()

TRAIN_DIR = ROOT_DIR / "dataset" / "train"
TEST_DIR = ROOT_DIR / "dataset" / "test"

EDA_DIR = ROOT_DIR / "eda_reports"
EDA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA LOADING
# ============================================================

def load_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    return pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        keep_default_na=False,
    )


print("=" * 80)
print("LOADING DATA")
print("=" * 80)

train_source1 = load_tsv(
    TRAIN_DIR / "train_source1.tsv"
)

train_source2 = load_tsv(
    TRAIN_DIR / "train_source2.tsv"
)

train_source3 = load_tsv(
    TRAIN_DIR / "train_source3.tsv"
)

ground_truth = load_tsv(
    TRAIN_DIR / "train_ground_truth.tsv"
)

test_source1 = load_tsv(
    TEST_DIR / "test_source1.tsv"
)

test_source2 = load_tsv(
    TEST_DIR / "test_source2.tsv"
)

test_source3 = load_tsv(
    TEST_DIR / "test_source3.tsv"
)


datasets = {
    "train_source1": train_source1,
    "train_source2": train_source2,
    "train_source3": train_source3,
    "test_source1": test_source1,
    "test_source2": test_source2,
    "test_source3": test_source3,
}


EXPECTED_COLUMNS = {
    "entity_id",
    "business_name",
    "business_address",
    "country",
}


# ============================================================
# SCHEMA VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("SCHEMA VALIDATION")
print("=" * 80)

for name, df in datasets.items():

    missing = EXPECTED_COLUMNS - set(df.columns)

    print(f"\n{name}")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    if missing:
        print("ERROR - Missing columns:", sorted(missing))
    else:
        print("Schema: OK")


print("\nGround truth")
print("Shape:", ground_truth.shape)
print("Columns:", ground_truth.columns.tolist())


# ============================================================
# BASIC DATASET SUMMARY
# ============================================================

summary_rows = []

for name, df in datasets.items():

    summary_rows.append({
        "dataset": name,
        "rows": len(df),
        "columns": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_entity_ids": int(
            df["entity_id"].duplicated().sum()
        ),
        "unique_entity_ids": int(
            df["entity_id"].nunique()
        ),
    })

dataset_summary = pd.DataFrame(summary_rows)

print("\n" + "=" * 80)
print("DATASET SUMMARY")
print("=" * 80)

print(
    dataset_summary.to_string(index=False)
)

dataset_summary.to_csv(
    EDA_DIR / "dataset_summary.tsv",
    sep="\t",
    index=False,
)


# ============================================================
# FIELD QUALITY
# ============================================================

field_rows = []

FIELDS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]


for dataset_name, df in datasets.items():

    for field in FIELDS:

        series = (
            df[field]
            .fillna("")
            .astype("string")
            .str.strip()
        )

        missing_mask = series.eq("")

        field_rows.append({
            "dataset": dataset_name,
            "field": field,
            "rows": len(df),
            "missing_count": int(
                missing_mask.sum()
            ),
            "missing_percentage": round(
                missing_mask.mean() * 100,
                4,
            ),
            "unique_values": int(
                series.nunique(dropna=False)
            ),
            "unique_percentage": round(
                series.nunique(dropna=False)
                / len(df)
                * 100,
                4,
            ),
        })


field_quality = pd.DataFrame(field_rows)

print("\n" + "=" * 80)
print("FIELD QUALITY")
print("=" * 80)

print(
    field_quality.to_string(index=False)
)

field_quality.to_csv(
    EDA_DIR / "field_quality.tsv",
    sep="\t",
    index=False,
)


# ============================================================
# DUPLICATE / COLLISION ANALYSIS
# ============================================================

def collision_stats(
    df: pd.DataFrame,
    column: str,
) -> dict:

    series = (
        df[column]
        .fillna("")
        .astype("string")
        .str.strip()
    )

    nonempty = series[series != ""]

    if len(nonempty) == 0:
        return {
            "nonempty_rows": 0,
            "unique_values": 0,
            "repeated_values": 0,
            "rows_in_collisions": 0,
            "collision_row_percentage": 0.0,
            "max_frequency": 0,
        }

    counts = nonempty.value_counts()

    repeated = counts[counts > 1]

    return {
        "nonempty_rows": int(len(nonempty)),
        "unique_values": int(len(counts)),
        "repeated_values": int(len(repeated)),
        "rows_in_collisions": int(repeated.sum()),
        "collision_row_percentage": round(
            repeated.sum()
            / len(nonempty)
            * 100,
            4,
        ),
        "max_frequency": int(counts.max()),
    }


print("\n" + "=" * 80)
print("RAW COLLISION ANALYSIS")
print("=" * 80)

collision_rows = []

for dataset_name, df in datasets.items():

    for field in [
        "business_name",
        "business_address",
        "country",
    ]:

        stats = collision_stats(
            df,
            field,
        )

        collision_rows.append({
            "dataset": dataset_name,
            "field": field,
            **stats,
        })


raw_collision_report = pd.DataFrame(
    collision_rows
)

print(
    raw_collision_report.to_string(index=False)
)

raw_collision_report.to_csv(
    EDA_DIR / "raw_collision_report.tsv",
    sep="\t",
    index=False,
)


# ============================================================
# COUNTRY DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("COUNTRY DISTRIBUTION")
print("=" * 80)

country_rows = []

for dataset_name, df in datasets.items():

    counts = (
        df["country"]
        .fillna("")
        .astype("string")
        .str.strip()
        .value_counts(dropna=False)
    )

    for country, count in counts.items():

        country_rows.append({
            "dataset": dataset_name,
            "country": country,
            "count": int(count),
            "percentage": round(
                count / len(df) * 100,
                4,
            ),
        })


country_report = pd.DataFrame(country_rows)

print(
    country_report
    .sort_values(
        ["dataset", "count"],
        ascending=[True, False],
    )
    .to_string(index=False)
)

country_report.to_csv(
    EDA_DIR / "country_distribution.tsv",
    sep="\t",
    index=False,
)


# ============================================================
# STRING LENGTH ANALYSIS
# ============================================================

length_rows = []

for dataset_name, df in datasets.items():

    name_length = (
        df["business_name"]
        .fillna("")
        .astype("string")
        .str.len()
    )

        address_length = (
            df["business_address"]
            .fillna("")
            .astype("string")
            .str.len()
        )

    name_words = (
        df["business_name"]
        .fillna("")
        .astype("string")
        .str.split()
        .str.len() # type: ignore
    )

    address_words = (
        df["business_address"]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len() # type: ignore
    )

    length_rows.extend([
        {
            "dataset": dataset_name,
            "field": "business_name",
            "metric": "characters_mean",
            "value": float(name_length.mean()),
        },
        {
            "dataset": dataset_name,
            "field": "business_name",
            "metric": "characters_median",
            "value": float(name_length.median()),
        },
        {
            "dataset": dataset_name,
            "field": "business_name",
            "metric": "characters_p99",
            "value": float(name_length.quantile(0.99)),
        },
        {
            "dataset": dataset_name,
            "field": "business_name",
            "metric": "words_mean",
            "value": float(name_words.mean()),
        },
        {
            "dataset": dataset_name,
            "field": "business_address",
            "metric": "characters_mean",
            "value": float(address_length.mean()),
        },
        {
            "dataset": dataset_name,
            "field": "business_address",
            "metric": "characters_median",
            "value": float(address_length.median()),
        },
        {
            "dataset": dataset_name,
            "field": "business_address",
            "metric": "characters_p99",
            "value": float(address_length.quantile(0.99)),
        },
        {
            "dataset": dataset_name,
            "field": "business_address",
            "metric": "words_mean",
            "value": float(address_words.mean()),
        },
    ])


length_report = pd.DataFrame(length_rows)

print("\n" + "=" * 80)
print("STRING LENGTH ANALYSIS")
print("=" * 80)

print(
    length_report.to_string(index=False)
)

length_report.to_csv(
    EDA_DIR / "length_report.tsv",
    sep="\t",
    index=False,
)


# ============================================================
# GROUND TRUTH PARSING
# ============================================================

print("\n" + "=" * 80)
print("GROUND TRUTH ANALYSIS")
print("=" * 80)


def split_match_ids(value: object) -> list[str]:
    if value is None or pd.isna(value): # type: ignore
        return []

    text = str(value).strip()

    if not text:
        return []

    return [
        item.strip()
        for item in text.split(",")
        if item.strip()
    ]


gt = ground_truth.copy()

gt["matched_ids_list"] = (
    gt["matched_entity_ids"]
    .apply(split_match_ids)
)

gt["num_matches"] = (
    gt["matched_ids_list"]
    .str.len()
)

print("\nMatch count distribution:")

print(
    gt["num_matches"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nMatch count statistics:")

print(
    gt["num_matches"].describe()
)


# ============================================================
# GROUND TRUTH VALIDATION
# ============================================================

source2_ids = set(
    train_source2["entity_id"].astype(str)
)

source3_ids = set(
    train_source3["entity_id"].astype(str)
)

source1_ids = set(
    train_source1["entity_id"].astype(str)
)

valid_target_ids = source2_ids | source3_ids


invalid_target_ids = []
duplicate_target_rows = []

for _, row in gt.iterrows():

    matched_ids = row["matched_ids_list"]

    invalid = [
        entity_id
        for entity_id in matched_ids
        if entity_id not in valid_target_ids
    ]

    if invalid:
        invalid_target_ids.append({
            "source1_entity_id": row["source1_entity_id"],
            "invalid_ids": ",".join(invalid),
        })

    if len(matched_ids) != len(set(matched_ids)):
        duplicate_target_rows.append(
            row["source1_entity_id"]
        )


print("\nGround truth validation")

print(
    "Invalid target IDs:",
    len(invalid_target_ids),
)

print(
    "Rows containing duplicate matched IDs:",
    len(duplicate_target_rows),
)

print(
    "Source 1 IDs missing from Source 1:",
    sum(
        str(x) not in source1_ids
        for x in gt["source1_entity_id"]
    ),
)


# ============================================================
# S2 / S3 MATCH DISTRIBUTION
# ============================================================

def classify_match_sources(
    matched_ids: list[str],
) -> str:

    has_s2 = any(
        entity_id in source2_ids
        for entity_id in matched_ids
    )

    has_s3 = any(
        entity_id in source3_ids
        for entity_id in matched_ids
    )

    if has_s2 and has_s3:
        return "S2_and_S3"

    if has_s2:
        return "S2_only"

    if has_s3:
        return "S3_only"

    if not matched_ids:
        return "No_match"

    return "Invalid"


gt["match_source_type"] = (
    gt["matched_ids_list"]
    .apply(classify_match_sources)
)

print("\nMatch source distribution:")

print(
    gt["match_source_type"]
    .value_counts()
    .to_string()
)


ground_truth_summary = (
    gt["num_matches"]
    .value_counts()
    .sort_index()
    .rename_axis("num_matches")
    .reset_index(name="source1_entities")
)

ground_truth_summary["percentage"] = (
    ground_truth_summary["source1_entities"]
    / len(gt)
    * 100
).round(4)

ground_truth_summary.to_csv(
    EDA_DIR / "ground_truth_match_distribution.tsv",
    sep="\t",
    index=False,
)

gt[
    [
        "source1_entity_id",
        "matched_entity_ids",
        "num_matches",
        "match_source_type",
    ]
].to_csv(
    EDA_DIR / "ground_truth_parsed.tsv",
    sep="\t",
    index=False,
)


# ============================================================
# PREPROCESS DATA
# ============================================================

print("\n" + "=" * 80)
print("RUNNING PREPROCESSING FOR EVALUATION")
print("=" * 80)

train_source1_processed = preprocess_dataframe(
    train_source1
)

train_source2_processed = preprocess_dataframe(
    train_source2
)

train_source3_processed = preprocess_dataframe(
    train_source3
)

test_source1_processed = preprocess_dataframe(
    test_source1
)

test_source2_processed = preprocess_dataframe(
    test_source2
)

test_source3_processed = preprocess_dataframe(
    test_source3
)


# ============================================================
# PREPROCESSING COLLISION ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("PROCESSED COLLISION ANALYSIS")
print("=" * 80)

processed_datasets = {
    "train_source1": train_source1_processed,
    "train_source2": train_source2_processed,
    "train_source3": train_source3_processed,
    "test_source1": test_source1_processed,
    "test_source2": test_source2_processed,
    "test_source3": test_source3_processed,
}


processed_collision_rows = []

for dataset_name, df in processed_datasets.items():

    for field in [
        "business_name_clean",
        "business_name_core",
        "business_name_compact",
        "business_address_clean",
        "address_compact",
    ]:

        stats = collision_stats(
            df,
            field,
        )

        processed_collision_rows.append({
            "dataset": dataset_name,
            "field": field,
            **stats,
        })


processed_collision_report = pd.DataFrame(
    processed_collision_rows
)

print(
    processed_collision_report.to_string(index=False)
)

processed_collision_report.to_csv(
    EDA_DIR / "processed_collision_report.tsv",
    sep="\t",
    index=False,
)


# ============================================================
# TRUE MATCH PAIRS
# ============================================================

print("\n" + "=" * 80)
print("TRUE MATCH PREPROCESSING EVALUATION")
print("=" * 80)


pairs = gt[
    [
        "source1_entity_id",
        "matched_ids_list",
    ]
].copy()

pairs = pairs.explode(
    "matched_ids_list"
)

pairs = pairs.rename(
    columns={
        "matched_ids_list": "matched_entity_id"
    }
)

pairs = pairs[
    pairs["matched_entity_id"]
    .fillna("")
    .astype(str)
    .str.strip()
    != ""
]


# Prefix columns so we can compare S1 and matched S2/S3.
s1_eval = (
    train_source1_processed
    .add_prefix("s1_")
)

s23_eval = pd.concat(
    [
        train_source2_processed.assign(match_source="S2"),
        train_source3_processed.assign(match_source="S3"),
    ],
    ignore_index=True,
)

s23_eval = s23_eval.add_prefix("m_")


pairs_eval = pairs.merge(
    s1_eval,
    left_on="source1_entity_id",
    right_on="s1_entity_id",
    how="left",
)

pairs_eval = pairs_eval.merge(
    s23_eval,
    left_on="matched_entity_id",
    right_on="m_entity_id",
    how="left",
)


print(
    "Number of known true match pairs:",
    len(pairs_eval),
)


# ============================================================
# EXACT AGREEMENT FUNCTIONS
# ============================================================

def exact_nonempty(
    left: pd.Series,
    right: pd.Series,
) -> pd.Series:

    left = (
        left.fillna("")
        .astype(str)
        .str.strip()
    )

    right = (
        right.fillna("")
        .astype(str)
        .str.strip()
    )

    return (
        (left != "")
        & (right != "")
        & (left == right)
    )


pairs_eval["raw_name_exact"] = exact_nonempty(
    pairs_eval["s1_business_name"],
    pairs_eval["m_business_name"],
)

pairs_eval["clean_name_exact"] = exact_nonempty(
    pairs_eval["s1_business_name_clean"],
    pairs_eval["m_business_name_clean"],
)

pairs_eval["core_name_exact"] = exact_nonempty(
    pairs_eval["s1_business_name_core"],
    pairs_eval["m_business_name_core"],
)

pairs_eval["ascii_name_exact"] = exact_nonempty(
    pairs_eval["s1_business_name_ascii"],
    pairs_eval["m_business_name_ascii"],
)

pairs_eval["raw_address_exact"] = exact_nonempty(
    pairs_eval["s1_business_address"],
    pairs_eval["m_business_address"],
)

pairs_eval["clean_address_exact"] = exact_nonempty(
    pairs_eval["s1_business_address_clean"],
    pairs_eval["m_business_address_clean"],
)

pairs_eval["ascii_address_exact"] = exact_nonempty(
    pairs_eval["s1_business_address_ascii"],
    pairs_eval["m_business_address_ascii"],
)

pairs_eval["numbers_exact"] = exact_nonempty(
    pairs_eval["s1_address_numbers"],
    pairs_eval["m_address_numbers"],
)

pairs_eval["country_exact"] = exact_nonempty(
    pairs_eval["s1_country_clean"],
    pairs_eval["m_country_clean"],
)

pairs_eval["clean_name_or_address_exact"] = (
    pairs_eval["clean_name_exact"]
    | pairs_eval["clean_address_exact"]
)

pairs_eval["clean_name_and_address_exact"] = (
    pairs_eval["clean_name_exact"]
    & pairs_eval["clean_address_exact"]
)


# ============================================================
# AGREEMENT REPORT
# ============================================================

agreement_fields = [
    "raw_name_exact",
    "clean_name_exact",
    "core_name_exact",
    "ascii_name_exact",
    "raw_address_exact",
    "clean_address_exact",
    "ascii_address_exact",
    "numbers_exact",
    "country_exact",
    "clean_name_or_address_exact",
    "clean_name_and_address_exact",
]


agreement_rows = []

for field in agreement_fields:

    exact_count = int(
        pairs_eval[field].sum()
    )

    agreement_rows.append({
        "measure": field,
        "exact_pairs": exact_count,
        "total_true_pairs": len(pairs_eval),
        "exact_percentage": round(
            exact_count
            / len(pairs_eval)
            * 100,
            4,
        ),
    })


agreement_report = pd.DataFrame(
    agreement_rows
)

print(
    agreement_report.to_string(index=False)
)

agreement_report.to_csv(
    EDA_DIR / "true_match_preprocessing_evaluation.tsv",
    sep="\t",
    index=False,
)


# ============================================================
# NORMALIZATION GAIN
# ============================================================

def percentage_for(
    report: pd.DataFrame,
    measure: str,
) -> float:

    row = report[
        report["measure"] == measure
    ]

    if row.empty:
        return 0.0

    return float(
        row.iloc[0]["exact_percentage"]
    )


raw_name = percentage_for(
    agreement_report,
    "raw_name_exact",
)

clean_name = percentage_for(
    agreement_report,
    "clean_name_exact",
)

core_name = percentage_for(
    agreement_report,
    "core_name_exact",
)

raw_address = percentage_for(
    agreement_report,
    "raw_address_exact",
)

clean_address = percentage_for(
    agreement_report,
    "clean_address_exact",
)


print("\nNormalization gains")

print(
    f"Name raw -> clean: "
    f"{clean_name - raw_name:+.4f} percentage points"
)

print(
    f"Name clean -> core: "
    f"{core_name - clean_name:+.4f} percentage points"
)

print(
    f"Address raw -> clean: "
    f"{clean_address - raw_address:+.4f} percentage points"
)


# ============================================================
# SHOW TRUE MATCH EXAMPLES RESCUED BY NORMALIZATION
# ============================================================

example_columns = [
    "source1_entity_id",
    "matched_entity_id",
    "s1_business_name",
    "m_business_name",
    "s1_business_name_clean",
    "m_business_name_clean",
    "s1_business_name_core",
    "m_business_name_core",
    "s1_business_address",
    "m_business_address",
    "s1_business_address_clean",
    "m_business_address_clean",
]


rescued_name_examples = pairs_eval[
    (~pairs_eval["raw_name_exact"])
    & pairs_eval["clean_name_exact"]
][example_columns].head(50)

rescued_core_examples = pairs_eval[
    (~pairs_eval["clean_name_exact"])
    & pairs_eval["core_name_exact"]
][example_columns].head(50)

rescued_address_examples = pairs_eval[
    (~pairs_eval["raw_address_exact"])
    & pairs_eval["clean_address_exact"]
][example_columns].head(50)


rescued_name_examples.to_csv(
    EDA_DIR / "name_normalization_rescued_examples.tsv",
    sep="\t",
    index=False,
)

rescued_core_examples.to_csv(
    EDA_DIR / "core_name_rescued_examples.tsv",
    sep="\t",
    index=False,
)

rescued_address_examples.to_csv(
    EDA_DIR / "address_normalization_rescued_examples.tsv",
    sep="\t",
    index=False,
)


print("\nExamples rescued by name normalization:")
print(
    rescued_name_examples.to_string(index=False)
)

print("\nExamples rescued by legal-suffix/core normalization:")
print(
    rescued_core_examples.to_string(index=False)
)

print("\nExamples rescued by address normalization:")
print(
    rescued_address_examples.to_string(index=False)
)


# ============================================================
# NORMALIZATION TRANSFORMATION AUDIT
# ============================================================

print("\n" + "=" * 80)
print("NORMALIZATION TRANSFORMATION AUDIT")
print("=" * 80)


transformation_rows = []

for dataset_name, original_df in {
    "train_source1": train_source1,
    "train_source2": train_source2,
    "train_source3": train_source3,
    "test_source1": test_source1,
    "test_source2": test_source2,
    "test_source3": test_source3,
}.items():

    processed_df = processed_datasets[
        dataset_name
    ]

    for i in range(len(original_df)):

        raw_name = str(
            original_df.iloc[i]["business_name"]
        )

        clean_name = str(
            processed_df.iloc[i]["business_name_clean"]
        )

        core_name = str(
            processed_df.iloc[i]["business_name_core"]
        )

        raw_address = str(
            original_df.iloc[i]["business_address"]
        )

        clean_address = str(
            processed_df.iloc[i]["business_address_clean"]
        )

        if (
            raw_name != clean_name
            or clean_name != core_name
            or raw_address != clean_address
        ):

            transformation_rows.append({
                "dataset": dataset_name,
                "entity_id": str(
                    original_df.iloc[i]["entity_id"]
                ),
                "raw_name": raw_name,
                "clean_name": clean_name,
                "core_name": core_name,
                "raw_address": raw_address,
                "clean_address": clean_address,
            })

        # We only need examples, not every transformed row.
        if len(transformation_rows) >= 5000:
            break


transformation_audit = pd.DataFrame(
    transformation_rows
)

transformation_audit.to_csv(
    EDA_DIR / "normalization_transformation_audit.tsv",
    sep="\t",
    index=False,
)

print(
    "Transformation examples saved:",
    len(transformation_audit),
)


# ============================================================
# VISUALIZATION 1:
# BUSINESS NAME LENGTH
# ============================================================

plt.figure(figsize=(10, 6))

for dataset_name, df in {
    "Source 1": train_source1,
    "Source 2": train_source2,
    "Source 3": train_source3,
}.items():

    values = (
        df["business_name"]
        .fillna("")
        .astype("string")
        .str.len()
    )

    upper = values.quantile(0.99)

    values = values[
        values <= upper
    ]

    plt.hist(
        values,
        bins=40,
        histtype="step",
        density=True,
        label=dataset_name,
    )

plt.title(
    "Business Name Length Distribution"
)

plt.xlabel("Characters")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()

plt.savefig(
    EDA_DIR / "business_name_length.png",
    dpi=150,
)

plt.close()


# ============================================================
# VISUALIZATION 2:
# ADDRESS LENGTH
# ============================================================

plt.figure(figsize=(10, 6))

for dataset_name, df in {
    "Source 1": train_source1,
    "Source 2": train_source2,
    "Source 3": train_source3,
}.items():

    values = (
        df["business_address"]
        .fillna("")
        .astype("string")
        .str.len()
    )

    upper = values.quantile(0.99)

    values = values[
        values <= upper
    ]

    plt.hist(
        values,
        bins=40,
        histtype="step",
        density=True,
        label=dataset_name,
    )

plt.title(
    "Business Address Length Distribution"
)

plt.xlabel("Characters")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()

plt.savefig(
    EDA_DIR / "business_address_length.png",
    dpi=150,
)

plt.close()


# ============================================================
# VISUALIZATION 3:
# MATCH COUNT
# ============================================================

match_counts = (
    ground_truth["num_matches"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(10, 6))

plt.bar(
    match_counts.index.astype(str),
    match_counts.to_numpy(),
)

plt.title(
    "Number of Matches per Source 1 Entity"
)

plt.xlabel("Number of matches")
plt.ylabel("Number of Source 1 entities")

plt.tight_layout()

plt.savefig(
    EDA_DIR / "ground_truth_match_counts.png",
    dpi=150,
)

plt.close()

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("EDA COMPLETE")
print("=" * 80)

print(
    f"All reports written to:\n{EDA_DIR}"
)

print("\nImportant files:")

for filename in [
    "dataset_summary.tsv",
    "field_quality.tsv",
    "raw_collision_report.tsv",
    "country_distribution.tsv",
    "ground_truth_match_distribution.tsv",
    "processed_collision_report.tsv",
    "true_match_preprocessing_evaluation.tsv",
    "name_normalization_rescued_examples.tsv",
    "core_name_rescued_examples.tsv",
    "address_normalization_rescued_examples.tsv",
]:

    print(
        f"  - {filename}"
    )