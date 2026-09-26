import os
import re
import pandas as pd



SOURCE1_PATH = "processed/train/source1_processed.tsv"
SOURCE2_PATH = "processed/train/source2_processed.tsv"

OUTPUT_PATH = "output/candidate_pairs.tsv"

ID_COL = "entity_id"


print("=" * 70)
print("LOADING DATA")
print("=" * 70)

source1 = pd.read_csv(
    SOURCE1_PATH,
    sep="\t",
    dtype=str
)

source2 = pd.read_csv(
    SOURCE2_PATH,
    sep="\t",
    dtype=str
)

print(f"Source 1 rows: {len(source1):,}")
print(f"Source 2 rows: {len(source2):,}")



REQUIRED_COLUMNS = [
    "entity_id",
    "country_clean",
    "business_name_clean",
    "business_name_core",
    "business_name_token_sorted",
    "business_address_clean",
    "business_address_token_sorted",
    "address_numbers",
]

for col in REQUIRED_COLUMNS:

    if col not in source1.columns:
        raise ValueError(
            f"Missing column in Source 1: {col}"
        )

    if col not in source2.columns:
        raise ValueError(
            f"Missing column in Source 2: {col}"
        )



def safe_string(series):
    """
    Convert NaN to empty strings and strip whitespace.
    """
    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )


for df in [source1, source2]:

    df["country_key"] = safe_string(
        df["country_clean"]
    )

    df["name_clean_key"] = safe_string(
        df["business_name_clean"]
    )

    df["name_core_key"] = safe_string(
        df["business_name_core"]
    )

    df["name_sorted_key"] = safe_string(
        df["business_name_token_sorted"]
    )

    df["address_clean_key"] = safe_string(
        df["business_address_clean"]
    )

    df["address_sorted_key"] = safe_string(
        df["business_address_token_sorted"]
    )

    df["address_numbers_key"] = safe_string(
        df["address_numbers"]
    )



LEGAL_SUFFIXES = {
    "pvt",
    "private",
    "limited",
    "ltd",
    "llp",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "company",
    "co",
}


def make_name_key(value):
    """
    Create a controlled name key for B6.

    We intentionally do NOT use only the first character(s).
    We use up to the first two meaningful tokens.
    """

    if not value:
        return ""

    tokens = value.split()

    tokens = [
        token
        for token in tokens
        if token not in LEGAL_SUFFIXES
    ]

    if not tokens:
        return ""

    return "_".join(tokens[:2])


for df in [source1, source2]:

    df["name_key"] = (
        df["name_core_key"]
        .apply(make_name_key)
    )


print("\nCreating blocking keys...")



for df in [source1, source2]:

    df["B1"] = (
        df["country_key"]
        + "||"
        + df["name_clean_key"]
    )



for df in [source1, source2]:

    df["B2"] = (
        df["country_key"]
        + "||"
        + df["name_core_key"]
    )



for df in [source1, source2]:

    df["B3"] = (
        df["country_key"]
        + "||"
        + df["name_sorted_key"]
    )



for df in [source1, source2]:

    df["B4"] = (
        df["country_key"]
        + "||"
        + df["address_clean_key"]
    )




for df in [source1, source2]:

    df["B5"] = (
        df["country_key"]
        + "||"
        + df["address_sorted_key"]
    )



for df in [source1, source2]:

    df["B6"] = ""

    valid = (
        df["country_key"].ne("")
        & df["address_numbers_key"].ne("")
        & df["name_key"].ne("")
    )

    df.loc[valid, "B6"] = (
        df.loc[valid, "country_key"]
        + "||"
        + df.loc[valid, "address_numbers_key"]
        + "||"
        + df.loc[valid, "name_key"]
    )



def generate_candidates(
    left,
    right,
    block_column,
    block_name
):

    print("\n" + "-" * 70)
    print(block_name)
    print("-" * 70)

    # Keep only useful keys.
    left_valid = left[
        left[block_column].ne("")
    ][
        [ID_COL, block_column]
    ].copy()

    right_valid = right[
        right[block_column].ne("")
    ][
        [ID_COL, block_column]
    ].copy()

    print(
        f"Valid S1 rows: {len(left_valid):,}"
    )

    print(
        f"Valid S2 rows: {len(right_valid):,}"
    )


    left_valid = left_valid.drop_duplicates()

    right_valid = right_valid.drop_duplicates()


    pairs = left_valid.merge(
        right_valid,
        on=block_column,
        how="inner",
        suffixes=("_s1", "_s2")
    )

    pairs = pairs[
        [f"{ID_COL}_s1", f"{ID_COL}_s2"]
    ]

    pairs.columns = [
        "source1_id",
        "source2_id"
    ]

    pairs = pairs.drop_duplicates()

    print(
        f"{block_name} candidates: "
        f"{len(pairs):,}"
    )

    return pairs


print("\n")
print("=" * 70)
print("GENERATING BLOCKS")
print("=" * 70)


B1 = generate_candidates(
    source1,
    source2,
    "B1",
    "B1 = country + clean name"
)


B2 = generate_candidates(
    source1,
    source2,
    "B2",
    "B2 = country + core name"
)


B3 = generate_candidates(
    source1,
    source2,
    "B3",
    "B3 = country + sorted name"
)


B4 = generate_candidates(
    source1,
    source2,
    "B4",
    "B4 = country + clean address"
)


B5 = generate_candidates(
    source1,
    source2,
    "B5",
    "B5 = country + sorted address"
)


B6 = generate_candidates(
    source1,
    source2,
    "B6",
    "B6 = country + address numbers + name key"
)



print("\n")
print("=" * 70)
print("CUMULATIVE UNION")
print("=" * 70)


blocks = [
    ("B1", B1),
    ("B2", B2),
    ("B3", B3),
    ("B4", B4),
    ("B5", B5),
    ("B6", B6),
]


candidate_sets = set()

for block_name, block_df in blocks:

    before = len(candidate_sets)

    # Convert each pair to tuple
    candidate_sets.update(
        zip(
            block_df["source1_id"],
            block_df["source2_id"]
        )
    )

    after = len(candidate_sets)

    print(
        f"{block_name}: "
        f"+{after - before:,} new pairs | "
        f"{after:,} cumulative pairs"
    )



candidate_pairs = pd.DataFrame(
    candidate_sets,
    columns=[
        "source1_id",
        "source2_id"
    ]
)



candidate_pairs = candidate_pairs.drop_duplicates(
    subset=[
        "source1_id",
        "source2_id"
    ]
)

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

candidate_pairs.to_csv(
    OUTPUT_PATH,
    sep="\t",
    index=False
)


print("\n")
print("=" * 70)
print("BLOCKING COMPLETE")
print("=" * 70)

print(
    f"Final candidate pairs: "
    f"{len(candidate_pairs):,}"
)

print(
    f"Saved to: {OUTPUT_PATH}"
)
