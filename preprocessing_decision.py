import os
import json
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_PATH = "networkTraffic.csv"
OUTPUT_DIR = "preprocessing_decision_output"

TARGET_COL = "attack_cat"
ID_COL = "id"
CATEGORICAL_COLS = ["proto", "service", "state"]
LIKELY_BINARY_COLS = ["is_ftp_login", "ct_ftp_cmd", "is_sm_ips_ports"]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    """
    Load the dataset and convert the assignment's missing-value marker '?' to NaN.
    """
    df = pd.read_csv(
        path,
        na_values=["?", "NA", "NaN", "null", "None", ""],
        keep_default_na=True,
    )
    df.columns = [c.strip() for c in df.columns]
    return df


def split_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Infer numeric vs categorical columns excluding target and identifier.
    """
    categorical = [c for c in CATEGORICAL_COLS if c in df.columns]
    numeric = [c for c in df.columns if c not in categorical + [TARGET_COL, ID_COL]]
    return numeric, categorical


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if miss.empty:
        return pd.DataFrame(columns=["column", "missing_count", "missing_pct"])
    return pd.DataFrame({
        "column": miss.index,
        "missing_count": miss.values,
        "missing_pct": (miss.values / len(df) * 100).round(2),
    })


def value_frequencies(df: pd.DataFrame, cols: List[str]) -> Dict[str, pd.DataFrame]:
    out = {}
    for col in cols:
        if col not in df.columns:
            continue
        vc = df[col].value_counts(dropna=False)
        out[col] = pd.DataFrame({
            "value": vc.index.astype(str),
            "count": vc.values,
            "pct": (vc.values / len(df) * 100).round(4),
        })
    return out


def numeric_profile(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    rows = []
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        rows.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "n_non_null": int(s.notna().sum()),
            "n_null": int(s.isna().sum()),
            "min": s.min(),
            "q1": s.quantile(0.25),
            "median": s.median(),
            "mean": s.mean(),
            "q3": s.quantile(0.75),
            "max": s.max(),
            "std": s.std(),
            "n_zero": int((s == 0).sum()),
            "n_negative": int((s < 0).sum()),
            "n_unique": int(s.nunique(dropna=True)),
        })
    return pd.DataFrame(rows)


def outlier_iqr_summary(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    rows = []
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 4:
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((s < lower) | (s > upper)).sum()
        rows.append({
            "column": col,
            "outlier_count": int(outliers),
            "outlier_pct": round(outliers / len(s) * 100, 2),
            "lower_bound": lower,
            "upper_bound": upper,
        })
    return pd.DataFrame(rows).sort_values(by="outlier_count", ascending=False)


def correlation_pairs(df: pd.DataFrame, numeric_cols: List[str], threshold: float = 0.95) -> pd.DataFrame:
    num_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    corr = num_df.corr(numeric_only=True)

    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if pd.notna(val) and abs(val) >= threshold:
                pairs.append({
                    "feature_1": cols[i],
                    "feature_2": cols[j],
                    "corr": val,
                })

    if not pairs:
        return pd.DataFrame(columns=["feature_1", "feature_2", "corr"])
    return pd.DataFrame(pairs).sort_values(by="corr", key=lambda s: s.abs(), ascending=False)


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COL not in df.columns:
        return pd.DataFrame(columns=["class", "count", "pct"])
    vc = df[TARGET_COL].value_counts(dropna=False)
    return pd.DataFrame({
        "class": vc.index.astype(str),
        "count": vc.values,
        "pct": (vc.values / len(df) * 100).round(2),
    })


def binary_anomaly_report(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        vals = sorted(pd.unique(s))
        rows.append({
            "column": col,
            "unique_values": vals,
            "value_counts": dict(pd.Series(s).value_counts().sort_index()),
            "note": "Inspect whether this feature should be treated as binary, count-like, or left numeric.",
        })
    return pd.DataFrame(rows)


def low_variance_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        rows.append({
            "column": col,
            "n_unique": int(df[col].nunique(dropna=True)),
            "is_constant": bool(df[col].nunique(dropna=True) == 1),
            "is_near_constant": bool(df[col].nunique(dropna=True) <= 2),
        })
    return pd.DataFrame(rows)


def make_recommendations(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, List[str]]:
    """
    Rule-based preprocessing recommendations for the two models.
    This is intentionally conservative and explainable for report writing.
    """
    rec = {
        "shared": [],
        "knn": [],
        "decision_tree": [],
    }

    # Shared
    if ID_COL in df.columns:
        rec["shared"].append("Remove 'id' because it is a unique identifier and not predictive.")
    if df[CATEGORICAL_COLS].isna().any().any():
        rec["shared"].append("Impute missing nominal values, especially 'service' (consider using a 'missing' category).")
    rec["shared"].append("Encode nominal features ('proto', 'service', 'state') before modelling.")
    rec["shared"].append("Use stratified cross-validation because the target distribution is strongly skewed.")
    rec["shared"].append("Report macro-F1 as the main metric, with accuracy as a secondary metric.")

    # kNN
    rec["knn"].append("Scale numeric features (StandardScaler or MinMaxScaler), because kNN is distance-based.")
    rec["knn"].append("Consider removing highly correlated/redundant features to avoid overweighting repeated information.")
    rec["knn"].append("Consider treating extreme outliers carefully (robust scaling or clipping) because kNN is sensitive to distance distortion.")
    rec["knn"].append("Remove constant/near-constant features if they do not add useful information.")

    # Tree
    rec["decision_tree"].append("Do not scale numeric features; decision trees are largely scale-invariant.")
    rec["decision_tree"].append("Keep original numeric values unless a value is clearly invalid.")
    rec["decision_tree"].append("Remove 'id' and any obviously useless constant features.")
    rec["decision_tree"].append("Tune tree depth and minimum leaf size to control overfitting, especially under class imbalance.")

    return rec


def print_section(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def save_dataframe(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)


def save_dict_text(d: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, default=str)


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    ensure_dir(OUTPUT_DIR)

    df = load_data(DATA_PATH)
    numeric_cols, categorical_cols = split_feature_types(df)

    # --------- Basic summaries ----------
    print_section("DATASET OVERVIEW")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("Columns:")
    print(list(df.columns))
    print("\nDtypes:")
    print(df.dtypes)
    print("\nHead:")
    print(df.head())

    print_section("MISSING VALUES")
    miss = missing_summary(df)
    print(miss if not miss.empty else "No missing values found.")

    print_section("CLASS DISTRIBUTION")
    cls = class_distribution(df)
    print(cls)

    print_section("CATEGORICAL VALUE FREQUENCIES")
    cat_freq = value_frequencies(df, categorical_cols + [TARGET_COL])
    for col, summary in cat_freq.items():
        print(f"\n--- {col} ---")
        print(summary.head(15))

    print_section("NUMERIC PROFILE")
    numprof = numeric_profile(df, numeric_cols)
    print(numprof.head(25))

    print_section("LOW VARIANCE / CONSTANT FEATURES")
    lv = low_variance_report(df)
    print(lv[lv["is_constant"] | lv["is_near_constant"]].sort_values(by="n_unique"))

    print_section("BINARY / COUNT ANOMALIES")
    binrep = binary_anomaly_report(df, LIKELY_BINARY_COLS)
    print(binrep)

    print_section("OUTLIERS (IQR RULE)")
    out = outlier_iqr_summary(df, numeric_cols)
    print(out.head(25))

    print_section("HIGH CORRELATION PAIRS")
    corr = correlation_pairs(df, numeric_cols, threshold=0.95)
    print(corr)

    print_section("PREPROCESSING RECOMMENDATIONS")
    rec = make_recommendations(df, numeric_cols)
    for group, items in rec.items():
        print(f"\n[{group.upper()}]")
        for item in items:
            print(f"- {item}")

    # --------- Save outputs ----------
    save_dataframe(miss, os.path.join(OUTPUT_DIR, "missing_values.csv"))
    save_dataframe(cls, os.path.join(OUTPUT_DIR, "class_distribution.csv"))
    save_dataframe(numprof, os.path.join(OUTPUT_DIR, "numeric_profile.csv"))
    save_dataframe(lv, os.path.join(OUTPUT_DIR, "low_variance_report.csv"))
    save_dataframe(binrep, os.path.join(OUTPUT_DIR, "binary_anomaly_report.csv"))
    save_dataframe(out, os.path.join(OUTPUT_DIR, "outlier_iqr_report.csv"))
    save_dataframe(corr, os.path.join(OUTPUT_DIR, "high_correlation_pairs.csv"))

    # Save categorical frequencies individually
    cat_dir = os.path.join(OUTPUT_DIR, "categorical_frequencies")
    ensure_dir(cat_dir)
    for col, summary in cat_freq.items():
        save_dataframe(summary, os.path.join(cat_dir, f"{col}_freq.csv"))

    save_dict_text(rec, os.path.join(OUTPUT_DIR, "preprocessing_recommendations.json"))

    # Also save a compact report file for easy reading
    report_lines = []
    report_lines.append("Preprocessing Decision Script Summary")
    report_lines.append(f"Dataset shape: {df.shape[0]} x {df.shape[1]}")
    report_lines.append("")
    report_lines.append("Recommended shared actions:")
    report_lines.extend([f"- {x}" for x in rec["shared"]])
    report_lines.append("")
    report_lines.append("Recommended kNN actions:")
    report_lines.extend([f"- {x}" for x in rec["knn"]])
    report_lines.append("")
    report_lines.append("Recommended Decision Tree actions:")
    report_lines.extend([f"- {x}" for x in rec["decision_tree"]])

    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print_section("DONE")
    print(f"Saved outputs to: {OUTPUT_DIR}")
    print("Files written:")
    print("- missing_values.csv")
    print("- class_distribution.csv")
    print("- numeric_profile.csv")
    print("- low_variance_report.csv")
    print("- binary_anomaly_report.csv")
    print("- outlier_iqr_report.csv")
    print("- high_correlation_pairs.csv")
    print("- preprocessing_recommendations.json")
    print("- summary.txt")


if __name__ == "__main__":
    main()