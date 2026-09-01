import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "model_results"
GRAPH_DIR = os.path.join(OUTPUT_DIR, "graphs")

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 11


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_csv(filename, **kwargs):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, **kwargs)


def plot_variant_comparison():
    df = load_csv("variant_comparison.csv")

    # Expected single-row optimized comparison; keep it flexible
    row = df.iloc[0]

    plot_df = pd.DataFrame([
        {"model": "kNN", "metric": "Macro-F1", "score": row["knn_macro_f1_mean"]},
        {"model": "kNN", "metric": "Accuracy", "score": row["knn_accuracy_mean"]},
        {"model": "Decision Tree", "metric": "Macro-F1", "score": row["tree_macro_f1_mean"]},
        {"model": "Decision Tree", "metric": "Accuracy", "score": row["tree_accuracy_mean"]},
    ])

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=plot_df, x="model", y="score", hue="metric", palette="Set2")
    ax.set_title("Model Comparison on Optimized Preprocessing")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(title="")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "model_comparison.png"))
    plt.close()


def plot_class_metric(report_file, title, filename, metric="f1-score"):
    df = load_csv(report_file, index_col=0)

    class_rows = [str(i) for i in range(10)]
    class_df = df.loc[class_rows, [metric]].reset_index()
    class_df.columns = ["class", metric]

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=class_df, x="class", y=metric, color="#4C72B0")
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel(metric.replace("-", " ").title())
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, filename))
    plt.close()


def load_confusion_matrix(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, header=0)


def normalize_confusion_matrix(cm):
    cm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return cm / row_sums


def plot_normalized_confusion_matrix(filename, title, outname):
    cm = load_confusion_matrix(filename).to_numpy()
    cm_norm = normalize_confusion_matrix(cm)

    plt.figure(figsize=(9, 7))
    ax = sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={"label": "Proportion"}
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, outname))
    plt.close()


def plot_support_distribution(report_file, title, filename):
    df = load_csv(report_file, index_col=0)
    class_rows = [str(i) for i in range(10)]
    class_df = df.loc[class_rows, ["support"]].reset_index()
    class_df.columns = ["class", "support"]

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=class_df, x="class", y="support", color="#55A868")
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Support")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, filename))
    plt.close()


def main():
    ensure_dir(GRAPH_DIR)

    # 1. Overall model comparison
    plot_variant_comparison()

    # 2. Class-wise metrics for optimized models
    plot_class_metric(
        "decision_tree_optimized_classification_report.csv",
        "Decision Tree Optimized: Per-Class F1-Score",
        "decision_tree_optimized_f1.png",
        metric="f1-score",
    )

    plot_class_metric(
        "knn_optimized_classification_report.csv",
        "kNN Optimized: Per-Class F1-Score",
        "knn_optimized_f1.png",
        metric="f1-score",
    )

    plot_class_metric(
        "decision_tree_optimized_classification_report.csv",
        "Decision Tree Optimized: Per-Class Recall",
        "decision_tree_optimized_recall.png",
        metric="recall",
    )

    plot_class_metric(
        "knn_optimized_classification_report.csv",
        "kNN Optimized: Per-Class Recall",
        "knn_optimized_recall.png",
        metric="recall",
    )

    # 3. Class support distribution
    plot_support_distribution(
        "decision_tree_optimized_classification_report.csv",
        "Class Support in the Evaluation Set",
        "class_support.png",
    )

    # 4. Normalized confusion matrices
    plot_normalized_confusion_matrix(
        "decision_tree_optimized_confusion_matrix.csv",
        "Decision Tree Optimized: Normalized Confusion Matrix",
        "decision_tree_optimized_cm_norm.png",
    )

    plot_normalized_confusion_matrix(
        "knn_optimized_confusion_matrix.csv",
        "kNN Optimized: Normalized Confusion Matrix",
        "knn_optimized_cm_norm.png",
    )

    print(f"Graphs saved to: {GRAPH_DIR}")


if __name__ == "__main__":
    main()