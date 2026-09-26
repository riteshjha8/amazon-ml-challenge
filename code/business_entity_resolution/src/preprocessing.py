from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd


# ============================================================
# FIND PROJECT ROOT
# ============================================================

def find_project_root() -> Path:
    """
    Find the project directory containing:
        dataset/train
        dataset/test
    """
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
PROCESSED_DIR = ROOT_DIR / "processed"


# ============================================================
# REQUIRED INPUT COLUMNS
# ============================================================

REQUIRED_COLUMNS = {
    "entity_id",
    "business_name",
    "business_address",
    "country",
}


# ============================================================
# NAME LEGAL-FORM ALIASES
# ============================================================

NAME_LEGAL_ALIASES = {
    "inc": "incorporated",
    "incorporated": "incorporated",

    "corp": "corporation",
    "corporation": "corporation",

    "co": "company",
    "company": "company",

    "ltd": "limited",
    "limited": "limited",

    "pvt": "private",
    "private": "private",

    "llc": "llc",
    "llp": "llp",
    "plc": "plc",

    "pte": "pte",
    "sarl": "sarl",
    "sas": "sas",
    "sasu": "sasu",
    "eurl": "eurl",
    "gmbh": "gmbh",
}


LEGAL_SUFFIXES = {
    "incorporated",
    "corporation",
    "company",
    "limited",
    "private",
    "llc",
    "llp",
    "plc",
    "pte",
    "sarl",
    "sas",
    "sasu",
    "eurl",
    "gmbh",
}


# ============================================================
# ADDRESS ALIASES
# ============================================================

ADDRESS_ALIASES = {
    "st": "street",
    "street": "street",

    "rd": "road",
    "road": "road",

    "ave": "avenue",
    "avenue": "avenue",

    "blvd": "boulevard",
    "boulevard": "boulevard",

    "ln": "lane",
    "lane": "lane",

    "dr": "drive",
    "drive": "drive",

    "ct": "court",
    "court": "court",

    "pl": "place",
    "place": "place",

    "hwy": "highway",
    "highway": "highway",

    "apt": "apartment",
    "apartment": "apartment",

    "flr": "floor",
    "floor": "floor",

    "ste": "suite",
    "suite": "suite",

    "bldg": "building",
    "building": "building",

    "nr": "near",
    "near": "near",

    "opp": "opposite",
    "opposite": "opposite",

    "no": "number",
    "number": "number",

    "n": "north",
    "north": "north",

    "s": "south",
    "south": "south",

    "e": "east",
    "east": "east",

    "w": "west",
    "west": "west",

    "pincode": "pin",
    "pin": "pin",
}


# ============================================================
# OUTPUT COLUMNS
# ============================================================

OUTPUT_COLUMNS = [
    # ID
    "entity_id",

    # Original values
    "business_name",
    "business_address",
    "country",

    # Business name representations
    "business_name_clean",
    "business_name_core",
    "business_name_compact",
    "business_name_ascii",
    "business_name_core_ascii",
    "business_name_token_sorted",

    # Address representations
    "business_address_clean",
    "address_compact",
    "business_address_ascii",
    "business_address_token_sorted",
    "address_compact",
    "business_address_ascii",
    "business_address_token_sorted",
    "address_numbers",

    # Country
    "country_clean",
]


# ============================================================
# BASIC TEXT NORMALIZATION
# ============================================================

def normalize_text(value: object) -> str:
    """
    Conservative generic text normalization.

    Keeps Unicode characters instead of destroying them.
    """
    if value is None or pd.isna(value): # type: ignore
        return ""

    text = str(value)

    # Unicode normalization.
    text = unicodedata.normalize("NFKC", text)

    # Case-insensitive normalization.
    text = text.casefold()

    # Common equivalent separators.
    text = text.replace("&", " and ")
    text = text.replace("/", " ")
    text = text.replace("-", " ")

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# TOKEN CLEANING
# ============================================================

def clean_tokens(value: object) -> list[str]:
    """
    Convert a field to cleaned Unicode-aware tokens.
    """
    text = normalize_text(value)

    if not text:
        return []

    tokens: list[str] = []

    for token in text.split():

        cleaned = "".join(
            ch
            for ch in token
            if ch.isalnum()
        )

        if cleaned:
            tokens.append(cleaned)

    return tokens


# ============================================================
# BUSINESS NAME NORMALIZATION
# ============================================================

def normalize_name(value: object) -> str:
    """
    Normalize business name while keeping legal forms.
    """
    return " ".join(
        clean_tokens(value)
    )


# ============================================================
# BUSINESS NAME CORE
# ============================================================

def strip_legal_suffixes(value: object) -> str:
    """
    Create a representation with trailing legal company forms
    removed.

    Example:

        ABC Motors Pvt Ltd
        ->
        abc motors
    """
    tokens = normalize_name(value).split()

    if not tokens:
        return ""

    # Canonicalize abbreviations first.
    canonical_tokens = [
        NAME_LEGAL_ALIASES.get(token, token)
        for token in tokens
    ]

    # Remove only trailing legal forms.
    while (
        len(canonical_tokens) > 1
        and canonical_tokens[-1] in LEGAL_SUFFIXES
    ):
        canonical_tokens.pop()

    return " ".join(canonical_tokens)


# ============================================================
# TOKEN-SORTED NAME
# ============================================================

def token_sorted_text(value: object) -> str:
    """
    Create an order-independent token representation.

    Example:

        "ABC STAR MOTORS"
        ->
        "abc motors star"

        "MOTORS ABC STAR"
        ->
        "abc motors star"

    IMPORTANT:
    This is an additional representation.
    We do not replace business_name_clean or business_name_core.
    """
    tokens = clean_tokens(value)

    if not tokens:
        return ""

    return " ".join(
        sorted(tokens)
    )


# ============================================================
# ADDRESS NORMALIZATION
# ============================================================

def normalize_address(value: object) -> str:
    """
    Normalize address while retaining information.
    """
    tokens = clean_tokens(value)

    normalized_tokens: list[str] = []

    for token in tokens:
        normalized_tokens.append(
            ADDRESS_ALIASES.get(token, token)
        )

    return " ".join(
        normalized_tokens
    )


# ============================================================
# TOKEN-SORTED ADDRESS
# ============================================================

def token_sorted_address(value: object) -> str:
    """
    Create an order-independent representation for addresses.

    Example:

        "12 Main Street Delhi"
        ->
        "12 delhi main street"

    This is only an additional representation.
    """
    tokens = clean_tokens(value)

    if not tokens:
        return ""

    normalized_tokens = [
        ADDRESS_ALIASES.get(token, token)
        for token in tokens
    ]

    return " ".join(
        sorted(normalized_tokens)
    )


# ============================================================
# REMOVE DIACRITICS
# ============================================================

def strip_diacritics(value: object) -> str:
    """
    Example:

        Café -> Cafe
    """
    if value is None or pd.isna(value): # type: ignore
        return ""

    text = str(value)

    decomposed = unicodedata.normalize(
        "NFKD",
        text,
    )

    return "".join(
        ch
        for ch in decomposed
        if not unicodedata.combining(ch)
    )


# ============================================================
# COMPACT REPRESENTATION
# ============================================================

def make_compact(value: object) -> str:
    """
    Remove spaces from an already normalized field.

    Example:

        "abc motors"
        ->
        "abcmotors"
    """
    if value is None or pd.isna(value): # type: ignore
        return ""

    return re.sub(
        r"\s+",
        "",
        str(value),
    )


# ============================================================
# ADDRESS NUMBER EXTRACTION
# ============================================================

def extract_numbers(value: object) -> str:
    """
    Extract all digit sequences from an address.

    Example:

        "12 Main Road Building 4"
        ->
        "12 4"
    """
    text = normalize_text(value)

    if not text:
        return ""

    numbers = re.findall(
        r"\d+",
        text,
    )

    return " ".join(numbers)


# ============================================================
# COUNTRY
# ============================================================

def normalize_country(value: object) -> str:
    """
    Conservative open-set country normalization.

    No hard-coded country filtering.
    """
    return normalize_text(value)


# ============================================================
# DATAFRAME PREPROCESSING
# ============================================================

def preprocess_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    # Validate columns.
    missing_columns = (
        REQUIRED_COLUMNS
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # Fill missing values.
    for column in REQUIRED_COLUMNS:
        df[column] = (
            df[column]
            .fillna("")
            .astype("string")
        )

    # --------------------------------------------------------
    # BUSINESS NAME
    # --------------------------------------------------------

    df["business_name_clean"] = (
        df["business_name"]
        .apply(normalize_name)
    )

    df["business_name_core"] = (
        df["business_name"]
        .apply(strip_legal_suffixes)
    )

    df["business_name_compact"] = (
        df["business_name_clean"]
        .apply(make_compact)
    )

    df["business_name_ascii"] = (
        df["business_name_clean"]
        .apply(strip_diacritics)
    )

    df["business_name_core_ascii"] = (
        df["business_name_core"]
        .apply(strip_diacritics)
    )

    # NEW:
    # Word-order independent representation.
    df["business_name_token_sorted"] = (
        df["business_name_clean"]
        .apply(token_sorted_text)
    )

    # --------------------------------------------------------
    # ADDRESS
    # --------------------------------------------------------

    df["business_address_clean"] = (
        df["business_address"]
        .apply(normalize_address)
    )

    df["address_compact"] = (
        df["business_address_clean"]
        .apply(make_compact)
    )

    df["business_address_ascii"] = (
        df["business_address_clean"]
        .apply(strip_diacritics)
    )

    # NEW:
    # Word-order independent address.
    df["business_address_token_sorted"] = (
        df["business_address"]
        .apply(token_sorted_address)
    )

    df["address_numbers"] = (
        df["business_address"]
        .apply(extract_numbers)
    )

    # --------------------------------------------------------
    # COUNTRY
    # --------------------------------------------------------

    df["country_clean"] = (
        df["country"]
        .apply(normalize_country)
    )

    return df[OUTPUT_COLUMNS]


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_source(
    input_path: Path,
    output_path: Path,
) -> None:

    print(
        f"\nLoading: {input_path}"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    df = pd.read_csv(
        input_path,
        sep="\t",
        dtype="string",
        keep_default_na=False,
    )

    print(
        f"Original shape: {df.shape}"
    )

    processed_df = preprocess_dataframe(
        df
    )
    processed_df = preprocess_dataframe(
        df
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed_df.to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    print(
        f"Processed shape: {processed_df.shape}"
    )

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    for split_name, split_dir, prefix in [
        ("train", TRAIN_DIR, "train"),
        ("test", TEST_DIR, "test"),
    ]:

        for source_name in [
            "source1",
            "source2",
            "source3",
        ]:

            input_path = (
                split_dir
                / f"{prefix}_{source_name}.tsv"
            )

            output_path = (
                PROCESSED_DIR
                / split_name
                / f"{source_name}_processed.tsv"
            )

            process_source(
                input_path,
                output_path,
            )


if __name__ == "__main__":
    main()