import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from constants import TRAIN_RATIO
def result_comparison():

    # Step 1: Find all result files
    files = glob.glob("src/results_*.csv")

    summary = []

    # Step 2: Read each file and compute averages
    for f in files:
        df = pd.read_csv(f)
        df = df[int(len(df) * TRAIN_RATIO):]  # Only consider the test set portion
        # Extract mode, emb_conf, top_n from filename
        parts = f.split("_")
        mode, emb_conf, top_n = parts[1], parts[2], parts[3].replace(".csv", "")
        
        avg_s2_rec = df["s2_rec"].mean()
        avg_s2_util = df["s2_util"].mean()
        avg_s2_prec = df["s2_prec"].mean()
        
        summary.append({
            "file": f,
            "mode": mode,
            "emb_conf": emb_conf,
            "top_n": top_n,
            "avg_s2_rec": avg_s2_rec,
            "avg_s2_util": avg_s2_util,
            "avg_s2_prec": avg_s2_prec
        })

    # Step 3: Create summary DataFrame
    summary_df = pd.DataFrame(summary)

    print(summary_df)

    # Step 4: Sort by mode (fixed first, then dynamic), then emb_conf, then top_n
    mode_order = {"fixed": 0, "dynamic": 1}
    summary_df["mode_order"] = summary_df["mode"].map(mode_order)
    summary_df = summary_df.sort_values(by=["mode_order", "emb_conf", "top_n"])

    # Step 5: Create custom labels for x-axis
    summary_df["label"] = summary_df.apply(
        lambda row: f"Mode: {row['mode']}, Emb Conf: {row['emb_conf']}, Top-n: {row['top_n']}",
        axis=1
    )

    # Step 6: Melt for plotting
    summary_melted = summary_df.melt(
        id_vars=["label","mode","emb_conf","top_n"],
        value_vars=["avg_s2_rec","avg_s2_prec","avg_s2_util"],
        var_name="metric",
        value_name="average"
    )

    # Step 7: Visualization
    plt.figure(figsize=(12,6))
    ax = sns.barplot(data=summary_melted, x="label", y="average", hue="metric")

    # Rotate x-ticks for readability
    plt.xticks(rotation=45, ha="right")

    # Add numeric labels (4 decimal places) on each bar
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:.4f}",
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=8, color='black', rotation=0)

    plt.title("Comparison of Average s2_rec, s2_prec, s2_util")
    plt.tight_layout()
    plt.savefig("src/comparison_results.png")
