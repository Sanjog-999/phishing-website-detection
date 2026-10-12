
"""
train_model.py
Usage:
    python train_model.py dataset.csv

Dataset CSV mein URL wala column aur label wala column hona chahiye.
Supported datasets:
  * Kaggle "Phishing Site URLs"  -> columns: URL, Label  (good / bad)
  * PhiUSIIL Phishing URL        -> columns: URL, label  (1 = legitimate, 0 = phishing)
  * Koi bhi aur CSV jisme url + label ho
"""
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from app import FEATURE_NAMES, extract_features

# ---- CONFIG ----------------------------------------------------------
# Agar label numeric (0/1) hai to batao kaunsi value phishing hai.
# PhiUSIIL mein 0 = phishing  -> PHISHING_NUMERIC_VALUE = 0
# Kuch datasets mein 1 = phishing -> isse 1 kar do
PHISHING_NUMERIC_VALUE = 0
MAX_ROWS_PER_CLASS = 60000  # training fast rakhne ke liye
# -----------------------------------------------------------------------

PHISHING_WORDS = {"bad", "phishing", "malicious", "malware", "defacement", "spam"}
LEGIT_WORDS = {"good", "legitimate", "legit", "benign", "safe"}


def find_column(df, candidates):
    lowered = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def to_binary(value):
    """Return 1 = phishing, 0 = legitimate."""
    s = str(value).strip().lower()
    if s in PHISHING_WORDS:
        return 1
    if s in LEGIT_WORDS:
        return 0
    try:
        return 1 if int(float(s)) == PHISHING_NUMERIC_VALUE else 0
    except ValueError:
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python train_model.py dataset.csv")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    url_col = find_column(df, ["url", "urls", "link", "website"])
    label_col = find_column(df, ["label", "labels", "class", "result", "status", "type"])
    if url_col is None or label_col is None:
        print("Columns mile:", list(df.columns))
        print("URL aur label columns detect nahi hue. train_model.py mein column names set karo.")
        sys.exit(1)

    df = df[[url_col, label_col]].dropna().drop_duplicates(subset=url_col)
    df["y"] = df[label_col].map(to_binary)
    df = df.dropna(subset=["y"])
    df["y"] = df["y"].astype(int)

    print("Class counts (1 = phishing, 0 = legit):")
    print(df["y"].value_counts(), "\n")
    if df["y"].nunique() < 2:
        print("Sirf ek hi class mili! Label mapping check karo (PHISHING_NUMERIC_VALUE).")
        sys.exit(1)

    # balance karo
    parts = [g.sample(min(len(g), MAX_ROWS_PER_CLASS), random_state=42) for _, g in df.groupby("y")]
    df = pd.concat(parts).sample(frac=1, random_state=42)

    print(f"Features nikal rahe hain {len(df)} URLs se ...")
    X = pd.DataFrame([extract_features(u) for u in df[url_col]], columns=FEATURE_NAMES)
    y = df["y"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print("\nAccuracy:", round(accuracy_score(y_test, pred), 4))
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, target_names=["legit", "phishing"]))

    joblib.dump(model, "phishing_model.pkl")
    joblib.dump(FEATURE_NAMES, "feature_names.pkl")
    print("Saved: phishing_model.pkl, feature_names.pkl")


if __name__ == "__main__":
    main()