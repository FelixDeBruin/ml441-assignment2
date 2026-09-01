import os
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# =========================
# CONFIGURATION
# =========================
DATA_PATH = "networkTraffic.csv"
OUTPUT_DIR = "model_results"
RANDOM_STATE = 42

TARGET_COL = "attack_cat"
ID_COL = "id"
CAT_COLS = ["proto", "service", "state"]

# Step A: small validation run
FAST_MODE = False
USE_FAST_SAMPLE = False
FAST_SAMPLE_SIZE = 50000
FAST_N_SPLITS = 3

# Step C: final report-friendly run
FINAL_N_SPLITS = 5

# Correlation threshold for removing redundant numeric features in kNN
CORR_THRESHOLD = 0.95

# Handle invalid binary values
CAP_BINARY_ANOMALIES = True

# Run both variants in fast validation; you can later set baseline False for final run if desired
RUN_BASELINE_VARIANT = False
RUN_OPTIMIZED_VARIANT = True

PREPROCESS_VARIANTS = [
    {
        "name": "baseline",
        "scale_numeric": True,
        "scaler_type": "robust",
        "drop_high_corr": False,
        "cap_binary": False,
    },
    {
        "name": "optimized",
        "scale_numeric": True,
        "scaler_type": "robust",
        "drop_high_corr": True,
        "cap_binary": True,
    },
]

# Smaller hyperparameter grids for fast validation
KNN_GRID_FAST = {
    "classifier__n_neighbors": [3, 5, 7],
    "classifier__weights": ["uniform", "distance"],
    "classifier__p": [2],
}

TREE_GRID_FAST = {
    "classifier__criterion": ["gini", "entropy"],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_split": [2, 10],
    "classifier__min_samples_leaf": [1, 5],
    "classifier__class_weight": [None, "balanced"],
}

# Slightly larger but still reasonable grids for final reporting
KNN_GRID_FINAL = {
    "classifier__n_neighbors": [3, 5, 7, 9, 11],
    "classifier__weights": ["uniform", "distance"],
    "classifier__p": [2],
}

TREE_GRID_FINAL = {
    "classifier__criterion": ["gini", "entropy"],
    "classifier__max_depth": [None, 10, 20, 30],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 5],
    "classifier__class_weight": [None, "balanced"],
}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def log(msg: str):
    print(msg, flush=True)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        na_values=["?", "NA", "NaN", "null", "None", ""],
        keep_default_na=True,
    )
    df.columns = [c.strip() for c in df.columns]
    return df


def preprocess_targets(df: pd.DataFrame):
    y = df[TARGET_COL]
    if y.dtype == "O" or str(y.dtype).startswith("string"):
        y = y.astype("category").cat.codes
    return y.astype(int)


def sample_stratified(df: pd.DataFrame, target_col: str, n_samples: int, random_state: int):
    if n_samples >= len(df):
        return df.copy()

    log(f"[SAMPLE] Creating stratified sample of {n_samples} rows for fast mode...")
    sampled_parts = []
    frac = n_samples / len(df)

    for _, group in df.groupby(target_col):
        take = max(1, int(round(len(group) * frac)))
        sampled_parts.append(group.sample(n=take, random_state=random_state))

    sampled = pd.concat(sampled_parts, axis=0).sample(frac=1.0, random_state=random_state)
    log(f"[SAMPLE] Sampled shape: {sampled.shape}")
    return sampled


def cap_binary_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["is_ftp_login", "ct_ftp_cmd"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].clip(lower=0, upper=1)
    return df


def find_high_corr_numeric_cols(df: pd.DataFrame, numeric_cols, threshold=0.95):
    num_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    corr = num_df.corr(numeric_only=True)

    to_drop = set()
    pairs = []

    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if pd.notna(val) and abs(val) >= threshold:
                c1, c2 = cols[i], cols[j]
                pairs.append((c1, c2, val))
                to_drop.add(c2)

    return sorted(to_drop), pairs


class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, drop_cols=None):
        self.drop_cols = drop_cols or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        cols = [c for c in self.drop_cols if c in X.columns]
        return X.drop(columns=cols, errors="ignore")


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool, scaler_type: str = "standard"):
    categorical_features = [c for c in CAT_COLS if c in X.columns]
    numeric_features = [c for c in X.columns if c not in categorical_features]

    if scale_numeric:
        if scaler_type == "robust":
            numeric_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
            ])
        else:
            numeric_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
    else:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(X: pd.DataFrame, drop_cols, scale_numeric: bool, scaler_type: str = "standard"):
    X_work = X.drop(columns=[c for c in drop_cols if c in X.columns], errors="ignore")
    preprocessor = build_preprocessor(X_work, scale_numeric=scale_numeric, scaler_type=scaler_type)

    return Pipeline([
        ("dropper", ColumnDropper(drop_cols=drop_cols)),
        ("preprocessor", preprocessor),
        ("classifier", None),
    ])


def build_model_grid(model_name: str, fast_mode: bool):
    if model_name == "knn":
        return KNN_GRID_FAST if fast_mode else KNN_GRID_FINAL
    if model_name == "tree":
        return TREE_GRID_FAST if fast_mode else TREE_GRID_FINAL
    raise ValueError("Unknown model name")


def evaluate_model(name: str, pipeline, param_grid, X, y, n_splits=10):
    log(f"\n{'=' * 100}")
    log(f"MODEL: {name}")
    log(f"{'=' * 100}")
    log(f"[{name}] Starting {n_splits}-fold grid search...")
    log(f"[{name}] Parameter grid size estimate: {np.prod([len(v) for v in param_grid.values()])} combinations")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=2,
        error_score="raise",
    )

    log(f"[{name}] Fitting grid search...")
    search.fit(X, y)
    log(f"[{name}] Grid search complete.")
    log(f"[{name}] Best parameters: {search.best_params_}")
    log(f"[{name}] Best mean CV macro-F1: {search.best_score_:.6f}")

    log(f"[{name}] Running cross-validation summary...")
    cv_results = cross_validate(
        search.best_estimator_,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )
    log(f"[{name}] Cross-validation summary complete.")

    summary = {
        "model": name,
        "best_params": search.best_params_,
        "best_cv_macro_f1_from_gridsearch": float(search.best_score_),
        "cv_accuracy_mean": float(np.mean(cv_results["test_accuracy"])),
        "cv_accuracy_std": float(np.std(cv_results["test_accuracy"])),
        "cv_f1_macro_mean": float(np.mean(cv_results["test_f1_macro"])),
        "cv_f1_macro_std": float(np.std(cv_results["test_f1_macro"])),
        "cv_f1_weighted_mean": float(np.mean(cv_results["test_f1_weighted"])),
        "cv_f1_weighted_std": float(np.std(cv_results["test_f1_weighted"])),
    }

    log(f"[{name}] Running out-of-fold predictions...")
    y_pred = cross_val_predict(search.best_estimator_, X, y, cv=cv, n_jobs=-1)
    log(f"[{name}] Out-of-fold prediction complete.")

    report = classification_report(y, y_pred, digits=4, output_dict=True)
    cm = confusion_matrix(y, y_pred)

    log(f"\n[{name}] Classification report:")
    log(classification_report(y, y_pred, digits=4))

    return summary, report, cm, search.best_estimator_


def save_results(name, summary, report, cm, output_dir):
    ensure_dir(output_dir)

    with open(os.path.join(output_dir, f"{name}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame(report).T.to_csv(os.path.join(output_dir, f"{name}_classification_report.csv"))
    pd.DataFrame(cm).to_csv(os.path.join(output_dir, f"{name}_confusion_matrix.csv"), index=False)


def run_variant(df: pd.DataFrame, variant: dict, fast_mode: bool, n_splits: int):
    log(f"\n{'#' * 100}")
    log(f"RUNNING VARIANT: {variant['name']}")
    log(f"{'#' * 100}")

    df_work = df.copy()

    if variant["cap_binary"]:
        log("[VARIANT] Capping binary anomalies in is_ftp_login and ct_ftp_cmd...")
        df_work = cap_binary_anomalies(df_work)

    y = preprocess_targets(df_work)
    X = df_work.drop(columns=[TARGET_COL])

    numeric_cols = [c for c in X.columns if c not in CAT_COLS + [ID_COL]]
    drop_corr_cols = []
    corr_pairs = []

    if variant["drop_high_corr"]:
        log("[VARIANT] Detecting and removing highly correlated numeric features...")
        drop_corr_cols, corr_pairs = find_high_corr_numeric_cols(X, numeric_cols, threshold=CORR_THRESHOLD)
        log(f"[VARIANT] High-correlation columns marked for drop: {drop_corr_cols}")

    log("[VARIANT] Building kNN pipeline...")
    knn_drop_cols = [ID_COL] + drop_corr_cols
    knn_pipeline = build_pipeline(
        X=X,
        drop_cols=knn_drop_cols,
        scale_numeric=variant["scale_numeric"],
        scaler_type=variant["scaler_type"],
    )
    knn_pipeline.steps[-1] = ("classifier", KNeighborsClassifier())

    knn_grid = build_model_grid("knn", fast_mode)
    knn_summ, knn_rep, knn_cm, _ = evaluate_model(
        f"kNN [{variant['name']}]",
        knn_pipeline,
        knn_grid,
        X,
        y,
        n_splits=n_splits,
    )
    save_results(f"knn_{variant['name']}", knn_summ, knn_rep, knn_cm, OUTPUT_DIR)

    log("[VARIANT] Building Decision Tree pipeline...")
    tree_drop_cols = [ID_COL]
    tree_pipeline = build_pipeline(
        X=X,
        drop_cols=tree_drop_cols,
        scale_numeric=False,
    )
    tree_pipeline.steps[-1] = ("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE))

    tree_grid = build_model_grid("tree", fast_mode)
    tree_summ, tree_rep, tree_cm, _ = evaluate_model(
        f"DecisionTree [{variant['name']}]",
        tree_pipeline,
        tree_grid,
        X,
        y,
        n_splits=n_splits,
    )
    save_results(f"decision_tree_{variant['name']}", tree_summ, tree_rep, tree_cm, OUTPUT_DIR)

    return {
        "variant": variant["name"],
        "high_corr_pairs": corr_pairs,
        "knn": knn_summ,
        "decision_tree": tree_summ,
    }


def main():
    log("[MAIN] Starting training script...")
    ensure_dir(OUTPUT_DIR)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find dataset at: {DATA_PATH}")

    log("[MAIN] Loading dataset...")
    df = load_data(DATA_PATH)
    log(f"[MAIN] Loaded dataset with shape {df.shape}")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found.")

    df = df[df[TARGET_COL].notna()].copy()

    if FAST_MODE and USE_FAST_SAMPLE:
        df = sample_stratified(df, TARGET_COL, FAST_SAMPLE_SIZE, RANDOM_STATE)

    log("[MAIN] Class distribution:")
    log(str(df[TARGET_COL].value_counts(dropna=False).sort_index()))

    variant_results = []

    if RUN_BASELINE_VARIANT:
        variant_results.append(
            run_variant(
                df=df,
                variant=PREPROCESS_VARIANTS[0],
                fast_mode=FAST_MODE,
                n_splits=FAST_N_SPLITS if FAST_MODE else FINAL_N_SPLITS,
            )
        )

    if RUN_OPTIMIZED_VARIANT:
        variant_results.append(
            run_variant(
                df=df,
                variant=PREPROCESS_VARIANTS[1],
                fast_mode=FAST_MODE,
                n_splits=FAST_N_SPLITS if FAST_MODE else FINAL_N_SPLITS,
            )
        )

    comparison_rows = []
    for r in variant_results:
        comparison_rows.append({
            "variant": r["variant"],
            "knn_macro_f1_mean": r["knn"]["cv_f1_macro_mean"],
            "knn_macro_f1_std": r["knn"]["cv_f1_macro_std"],
            "knn_accuracy_mean": r["knn"]["cv_accuracy_mean"],
            "tree_macro_f1_mean": r["decision_tree"]["cv_f1_macro_mean"],
            "tree_macro_f1_std": r["decision_tree"]["cv_f1_macro_std"],
            "tree_accuracy_mean": r["decision_tree"]["cv_accuracy_mean"],
        })

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(os.path.join(OUTPUT_DIR, "variant_comparison.csv"), index=False)

    log("\n" + "=" * 100)
    log("VARIANT COMPARISON")
    log("=" * 100)
    log(str(comparison))

    best_choice = None
    best_score = -1

    for r in variant_results:
        for model_name in ["knn", "decision_tree"]:
            score = r[model_name]["cv_f1_macro_mean"]
            if score > best_score:
                best_score = score
                best_choice = (r["variant"], model_name, score)

    with open(os.path.join(OUTPUT_DIR, "best_overall_model.txt"), "w", encoding="utf-8") as f:
        f.write(f"Best overall model: variant={best_choice[0]}, model={best_choice[1]}, macro_f1={best_choice[2]:.6f}\n")

    log("\n[MAIN] Best overall model:")
    log(f"variant={best_choice[0]}, model={best_choice[1]}, macro_f1={best_choice[2]:.6f}")
    log(f"[MAIN] All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()