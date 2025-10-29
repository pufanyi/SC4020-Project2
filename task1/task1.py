# -----------------------------------------------
# Task 1: Symptom Co-occurrence Pattern Analysis
# Using Apriori Algorithm for Disease-Symptom Dataset
# -----------------------------------------------

import re

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Load the dataset
df = pd.read_csv(r"C:\Users\13248\Desktop\4020 project2\Dataset.csv")
print("Original dataset preview:")
print(df.head())

# Data Cleaning - normalize column names and text
# Strip column names
df.columns = [c.strip() for c in df.columns]

# Identify all symptom columns (Symptom_1 ... Symptom_17)
symptom_cols = [c for c in df.columns if "symptom" in c.lower()]


def normalize_symptom_name(s: str) -> str:
    """
    Clean and normalize a symptom string:
    - lowercase
    - trim spaces
    - fix inconsistent underscores / spaces
    - merge known synonyms
    """
    if not isinstance(s, str):
        return ""

    # lowercase and trim
    s = s.strip().lower()

    # replace multiple spaces with single space
    s = re.sub(r"\s+", " ", s)

    # fix things like "foul_smell_of urine" -> "foul_smell_of_urine"
    s = s.replace(" ", "_")

    # remove accidental double underscores like "spotting__urination"
    s = re.sub(r"_+", "_", s)

    # known synonym/unification map
    synonym_map = {
        # unify abdominal / belly / stomach pain
        "belly_pain": "abdominal_pain",
        "stomach_pain": "abdominal_pain",
        # minor typos / spacing normalization
        "foul_smell_of_urine": "foul_smell_of_urine",
        "spotting_urination": "spotting_urination",
    }

    # after normalization above, apply mapping
    if s in synonym_map:
        s = synonym_map[s]

    # also handle explicit special cases we saw in raw data
    # (in case original had space like "foul_smell_of urine" etc.)
    if s == "foul_smell_of_urine":
        s = "foul_smell_of_urine"
    if s == "spotting_urination":
        s = "spotting_urination"

    # return empty if it's something like "nan"
    if s == "nan":
        return ""

    return s


# apply cleaning to every symptom column
for c in symptom_cols:
    df[c] = df[c].astype(str).apply(normalize_symptom_name)

print("After normalization (preview):")
print(df.head())

# Generate transactions
# Each disease is treated as one "basket" containing multiple symptoms.
transactions = []
for _, row in df.iterrows():
    symptoms = [
        s
        for s in row[symptom_cols].tolist()
        if s != ""  # drop blanks
    ]
    # deduplicate symptoms inside the same disease record
    symptoms = list(set(symptoms))
    if len(symptoms) > 0:
        transactions.append(symptoms)

print("\nExample transactions after normalization (first 5):")
for t in transactions[:5]:
    print(t)

# One-Hot Encoding
# Convert the list of transactions into a 0/1 matrix suitable for Apriori algorithm
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

print("\nMatrix shape:", df_encoded.shape)
print(df_encoded.head())

# Frequent Itemset Mining using Apriori
# min_support = 0.05 means the combination appears in at least 5% of the diseases
frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)
frequent_itemsets = frequent_itemsets.sort_values(by="support", ascending=False)

print("\n Top 10 Frequent Symptom Combinations:")
print(frequent_itemsets.head(10))

# Generate Association Rules
# Rules represent relationships between symptoms (e.g. A → B)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules = rules.sort_values(by=["lift", "confidence"], ascending=[False, False])

print("\n Top 10 Association Rules:")
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10))

# Save the results
frequent_itemsets.to_csv("frequent_symptom_sets.csv", index=False)
rules.to_csv("association_rules.csv", index=False)
print("\n Results exported to 'frequent_symptom_sets.csv' and 'association_rules.csv'")
