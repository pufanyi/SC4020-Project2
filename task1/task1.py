"""Task 1: Symptom co-occurrence pattern analysis with the Apriori algorithm."""

import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "dataset.csv"
RESULTS_DIR = BASE_DIR / "results" / "task1"
MIN_SUPPORT = 0.05
CONFIDENCE_THRESHOLD = 0.5
FREQUENT_ITEMSETS_PATH = RESULTS_DIR / "frequent_symptom_sets.csv"
ASSOCIATION_RULES_PATH = RESULTS_DIR / "association_rules.csv"

SYNONYM_MAP = {
    "belly_pain": "abdominal_pain",
    "stomach_pain": "abdominal_pain",
    "foul_smell_of_urine": "foul_smell_of_urine",
    "spotting_urination": "spotting_urination",
}


def load_dataset(path: Path) -> pd.DataFrame:
    """Read the raw dataset and standardise column names."""
    df = pd.read_csv(path)
    print("Original dataset preview:")
    print(df.head())
    df.columns = [column.strip() for column in df.columns]
    return df


def identify_symptom_columns(df: pd.DataFrame) -> list[str]:
    """Return the columns that contain symptom information."""
    return [column for column in df.columns if "symptom" in column.lower()]


def normalize_symptom_name(raw: str) -> str:
    """Clean, normalise, and standardise the symptom labels."""
    if not isinstance(raw, str):
        return ""
    symptom = raw.strip().lower()
    symptom = re.sub(r"\s+", " ", symptom)
    symptom = symptom.replace(" ", "_")
    symptom = re.sub(r"_+", "_", symptom)
    symptom = SYNONYM_MAP.get(symptom, symptom)
    return "" if symptom == "nan" else symptom


def normalize_symptom_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Apply the symptom name normalisation to the selected columns."""
    for column in columns:
        df[column] = df[column].astype(str).map(normalize_symptom_name)
    print("After normalization (preview):")
    print(df.head())


def generate_transactions(df: pd.DataFrame, columns: Iterable[str]) -> list[list[str]]:
    """Create a transaction list with the symptoms present for each disease."""
    transactions: list[list[str]] = []
    for _, row in df[list(columns)].iterrows():
        symptoms = [symptom for symptom in row.tolist() if symptom]
        symptoms = list(set(symptoms))
        if symptoms:
            transactions.append(symptoms)
    return transactions


def encode_transactions(transactions: list[list[str]]) -> pd.DataFrame:
    """One-hot encode the transactions for use with Apriori."""
    encoder = TransactionEncoder()
    encoded_array = encoder.fit(transactions).transform(transactions)
    return pd.DataFrame(encoded_array, columns=encoder.columns_)


def mine_frequent_itemsets(df_encoded: pd.DataFrame) -> pd.DataFrame:
    """Run the Apriori algorithm and order the results by support."""
    itemsets = apriori(df_encoded, min_support=MIN_SUPPORT, use_colnames=True)
    return itemsets.sort_values(by="support", ascending=False)


def compute_association_rules(frequent_itemsets: pd.DataFrame) -> pd.DataFrame:
    """Compute and order association rules derived from the frequent itemsets."""
    rules = association_rules(
        frequent_itemsets, metric="confidence", min_threshold=CONFIDENCE_THRESHOLD
    )
    return rules.sort_values(by=["lift", "confidence"], ascending=[False, False])


def save_results(frequent_itemsets: pd.DataFrame, rules: pd.DataFrame) -> None:
    """Persist the itemsets and rules to disk."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frequent_itemsets.to_csv(FREQUENT_ITEMSETS_PATH, index=False)
    rules.to_csv(ASSOCIATION_RULES_PATH, index=False)
    print(
        f"\n Results exported to '{FREQUENT_ITEMSETS_PATH}' "
        f"and '{ASSOCIATION_RULES_PATH}'."
    )


def main() -> None:
    """Execute the end-to-end Apriori workflow."""
    df = load_dataset(DATA_PATH)
    symptom_columns = identify_symptom_columns(df)
    normalize_symptom_columns(df, symptom_columns)

    transactions = generate_transactions(df, symptom_columns)
    print("\nExample transactions after normalization (first 5):")
    for transaction in transactions[:5]:
        print(transaction)

    df_encoded = encode_transactions(transactions)
    print("\nMatrix shape:", df_encoded.shape)
    print(df_encoded.head())

    frequent_itemsets = mine_frequent_itemsets(df_encoded)
    print("\n Top 10 Frequent Symptom Combinations:")
    print(frequent_itemsets.head(10))

    rules = compute_association_rules(frequent_itemsets)
    print("\n Top 10 Association Rules:")
    print(
        rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10)
    )

    save_results(frequent_itemsets, rules)


if __name__ == "__main__":
    main()
