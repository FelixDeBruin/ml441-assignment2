import os
import re
import json
import warnings
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


DATA_PATH = "networkTraffic.csv"
OUTPUT_DIR = "dataset_report"


def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the dataset without forcing premature type conversion.
    We want to inspect raw values first because some columns contain '?'
    or mixed numeric/string values.
    """
    df = pd.read_csv(path, na_values=["?", "NA", "NaN", "null", "None", ""], keep_default_na=True)
    return df


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip whitespace from column names and make them consistent.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def print_basic_info(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("BASIC DATASET INFORMATION")
    print("=" * 80)
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nColumns:")
    print(list(df.columns))

    print("\nDtypes:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nLast 5 rows:")
    print(df.tail())


def find_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize missing values per column.
    """
    miss = df.isna().sum().sort_values(ascending=False)
    miss_pct = (miss / len(df) * 100).round(2)
    summary = pd.DataFrame({"missing_count": miss, "missing_pct": miss_pct})
    summary = summary[summary["missing_count"] > 0]
    return summary


def detect_string_numeric_issues(df: pd.DataFrame, numeric_candidates):
    """
    For columns that should be numeric, detect values that cannot be converted.
    """
    issues = []

    for col in numeric_candidates:
        if col not in df.columns:
            continue
        series = df[col]

        # Work only on non-null strings/values
        non_null = series.dropna().astype(str)

        coerced = pd.to_numeric(non_null, errors="coerce")
        bad_mask = coerced.isna()

        bad_values = non_null[bad_mask].unique().tolist()
        if len(bad_values) > 0:
            issues.append({
                "column": col,
                "bad_count": int(bad_mask.sum()),
                "unique_bad_values": bad_values[:20]
            })

    return pd.DataFrame(issues)


def infer_feature_types(df: pd.DataFrame):
    """
    Infer likely numeric and categorical columns based on the assignment description
    and observed data. We also keep a target column separate.
    """
    target_col = "attack_cat"

    categorical_cols = [c for c in ["proto", "service", "state", target_col] if c in df.columns]

    # Everything else except id and target is likely numeric
    exclude = set(categorical_cols + ["id"])
    numeric_cols = [c for c in df.columns if c not in exclude]

    return numeric_cols, categorical_cols, target_col


def summarize_categorical(df: pd.DataFrame, categorical_cols):
    """
    Show cardinality and most frequent values.
    """
    rows = []
    for col in categorical_cols:
        if col not in df.columns:
            continue
        vc = df[col].value_counts(dropna=False)
        rows.append({
            "column": col,
            "n_unique_including_na": df[col].nunique(dropna=False),
            "n_unique_excluding_na": df[col].nunique(dropna=True),
            "top_value": vc.index[0],
            "top_count": int(vc.iloc[0]),
            "top_pct": round(vc.iloc[0] / len(df) * 100, 2)
        })
    return pd.DataFrame(rows)


def summarize_numeric(df: pd.DataFrame, numeric_cols):
    """
    Basic numeric summary with coercion for columns that may contain mixed values.
    """
    numeric_summary = []
    for col in numeric_cols:
        if col not in df.columns:
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        numeric_summary.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "n_non_null": int(coerced.notna().sum()),
            "min": coerced.min(),
            "q1": coerced.quantile(0.25),
            "median": coerced.median(),
            "mean": coerced.mean(),
            "q3": coerced.quantile(0.75),
            "max": coerced.max(),
            "std": coerced.std(),
            "n_zero": int((coerced == 0).sum()),
            "n_negative": int((coerced < 0).sum())
        })
    return pd.DataFrame(numeric_summary)


def detect_constant_and_low_variance(df: pd.DataFrame, threshold=1):
    """
    Detect constant features and near-constant features.
    """
    rows = []
    for col in df.columns:
        nunique = df[col].nunique(dropna=True)
        rows.append({
            "column": col,
            "n_unique": nunique,
            "constant": nunique == 1,
            "near_constant": nunique <= threshold + 1
        })
    return pd.DataFrame(rows)


def detect_binary_anomalies(df: pd.DataFrame):
    """
    Detect columns that look binary but contain values outside {0,1}.
    """
    rows = []
    for col in df.columns:
        series = df[col].dropna()
        if len(series) == 0:
            continue

        # Try to identify binary-like columns
        unique_vals = sorted(pd.unique(series))
        if len(unique_vals) <= 5:
            numeric = pd.to_numeric(series, errors="coerce")
            vals = sorted(pd.unique(numeric.dropna()))
            if len(vals) > 0:
                if set(vals).issubset({0, 1}):
                    continue
                if any(v not in {0, 1, np.nan} for v in vals):
                    rows.append({
                        "column": col,
                        "unique_values": vals,
                        "note": "Potential binary feature with unexpected values"
                    })
    return pd.DataFrame(rows)


def outlier_summary(df: pd.DataFrame, numeric_cols):
    """
    Use the IQR rule to count potential outliers per numeric column.
    """
    rows = []
    for col in numeric_cols:
        if col not in df.columns:
            continue
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
            "upper_bound": upper
        })
    return pd.DataFrame(rows).sort_values(by="outlier_count", ascending=False)


def correlation_summary(df: pd.DataFrame, numeric_cols, threshold=0.95):
    """
    Find highly correlated numeric features.
    """
    num_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    corr = num_df.corr(numeric_only=True)

    high_corr_pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if pd.notna(val) and abs(val) >= threshold:
                high_corr_pairs.append({
                    "feature_1": cols[i],
                    "feature_2": cols[j],
                    "corr": val
                })

    return corr, pd.DataFrame(high_corr_pairs).sort_values(by="corr", key=lambda s: s.abs(), ascending=False)


def class_distribution(df: pd.DataFrame, target_col="attack_cat"):
    if target_col not in df.columns:
        return pd.DataFrame()
    vc = df[target_col].value_counts(dropna=False)
    out = pd.DataFrame({
        "class": vc.index.astype(str),
        "count": vc.values,
        "pct": (vc.values / len(df) * 100).round(2)
    })
    return out


def save_outputs(df, numeric_cols, categorical_cols, target_col):
    ensure_output_dir(OUTPUT_DIR)

    # Missing values
    miss = find_missing_values(df)
    miss.to_csv(os.path.join(OUTPUT_DIR, "missing_values.csv"))

    # Categoricals
    cat_summary = summarize_categorical(df, categorical_cols)
    cat_summary.to_csv(os.path.join(OUTPUT_DIR, "categorical_summary.csv"), index=False)

    # Numeric summary
    num_summary = summarize_numeric(df, numeric_cols)
    num_summary.to_csv(os.path.join(OUTPUT_DIR, "numeric_summary.csv"), index=False)

    # Constant / low variance
    const_summary = detect_constant_and_low_variance(df)
    const_summary.to_csv(os.path.join(OUTPUT_DIR, "constant_low_variance.csv"), index=False)

    # Binary anomalies
    bin_anoms = detect_binary_anomalies(df)
    bin_anoms.to_csv(os.path.join(OUTPUT_DIR, "binary_anomalies.csv"), index=False)

    # Outliers
    outliers = outlier_summary(df, numeric_cols)
    outliers.to_csv(os.path.join(OUTPUT_DIR, "outlier_summary.csv"), index=False)

    # Correlations
    corr, high_corr = correlation_summary(df, numeric_cols)
    corr.to_csv(os.path.join(OUTPUT_DIR, "correlation_matrix.csv"))
    high_corr.to_csv(os.path.join(OUTPUT_DIR, "high_correlations.csv"), index=False)

    # Class distribution
    dist = class_distribution(df, target_col)
    dist.to_csv(os.path.join(OUTPUT_DIR, "class_distribution.csv"), index=False)

    # Full schema snapshot
    schema = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(t) for t in df.dtypes]
    })
    schema.to_csv(os.path.join(OUTPUT_DIR, "schema.csv"), index=False)


def print_interpretive_notes(df, numeric_cols, categorical_cols, target_col):
    print("\n" + "=" * 80)
    print("PREPROCESSING IMPLICATIONS")
    print("=" * 80)

    print("\nLikely required for BOTH models:")
    print("- Remove 'id' because it is a unique identifier and not predictive.")
    print("- Handle missing values represented by '?' / NaN.")
    print("- Inspect and possibly correct invalid categorical/numeric values.")
    print("- Encode nominal categorical features: proto, service, state.")
    print("- Examine the skewed target distribution.")

    print("\nLikely required for kNN:")
    print("- Scale numeric features (e.g. StandardScaler or MinMaxScaler).")
    print("- Consider removing irrelevant / constant / near-constant features.")
    print("- Consider whether outliers should be capped or robust-scaled.")

    print("\nLikely required for Decision Trees:")
    print("- Scaling is not necessary.")
    print("- Retain numeric features in original scale unless there is a compelling reason.")
    print("- Missing values still need handling.")
    print("- Remove 'id' and any clearly useless constant features.")

    print("\nPotential issues visible in this dataset sample:")
    if "service" in df.columns:
        print(f"- service unique values (including missing): {df['service'].nunique(dropna=False)}")
    if "proto" in df.columns:
        print(f"- proto unique values (including missing): {df['proto'].nunique(dropna=False)}")
    if "state" in df.columns:
        print(f"- state unique values (including missing): {df['state'].nunique(dropna=False)}")

    if target_col in df.columns:
        print(f"- target classes (including missing): {df[target_col].nunique(dropna=False)}")


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find dataset at: {DATA_PATH}")

    df = load_dataset(DATA_PATH)
    df = normalize_column_names(df)

    numeric_cols, categorical_cols, target_col = infer_feature_types(df)

    print_basic_info(df)

    print("\n" + "=" * 80)
    print("MISSING VALUES SUMMARY")
    print("=" * 80)
    miss = find_missing_values(df)
    print(miss if not miss.empty else "No missing values found.")

    print("\n" + "=" * 80)
    print("CATEGORICAL FEATURES SUMMARY")
    print("=" * 80)
    cat_summary = summarize_categorical(df, categorical_cols)
    print(cat_summary if not cat_summary.empty else "No categorical summaries available.")

    print("\n" + "=" * 80)
    print("NUMERIC FEATURES SUMMARY")
    print("=" * 80)
    num_summary = summarize_numeric(df, numeric_cols)
    print(num_summary.head(20))

    print("\n" + "=" * 80)
    print("CONSTANT / NEAR-CONSTANT FEATURES")
    print("=" * 80)
    const_summary = detect_constant_and_low_variance(df)
    print(const_summary[const_summary["constant"] | const_summary["near_constant"]].sort_values(by="n_unique"))

    print("\n" + "=" * 80)
    print("BINARY ANOMALIES")
    print("=" * 80)
    bin_anoms = detect_binary_anomalies(df)
    print(bin_anoms if not bin_anoms.empty else "No obvious binary anomalies found.")

    print("\n" + "=" * 80)
    print("OUTLIER SUMMARY (TOP 20)")
    print("=" * 80)
    outliers = outlier_summary(df, numeric_cols)
    print(outliers.head(20))

    print("\n" + "=" * 80)
    print("HIGH CORRELATION PAIRS")
    print("=" * 80)
    _, high_corr = correlation_summary(df, numeric_cols)
    print(high_corr.head(50) if not high_corr.empty else "No very high correlations found using the selected threshold.")

    print("\n" + "=" * 80)
    print("TARGET CLASS DISTRIBUTION")
    print("=" * 80)
    dist = class_distribution(df, target_col)
    print(dist if not dist.empty else f"Target column '{target_col}' not found.")

    print_interpretive_notes(df, numeric_cols, categorical_cols, target_col)

    save_outputs(df, numeric_cols, categorical_cols, target_col)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Saved summary files to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()