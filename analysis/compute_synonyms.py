import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from data import display_train


def plot_synonym_scores(synonym_scores: pd.DataFrame):
    # Plot Evaluation Dataset Scores
    synonym_scores = (
        synonym_scores[["is_synonym", "train_dataset", "metric", "value"]]
        .groupby(["is_synonym", "train_dataset", "metric"])
        .agg({"value": ["mean", "std"]})
        .reset_index()
    )
    synonym_scores.columns = ["Is Synonym", "FT-Dataset", "Metric", "mean", "std"]
    synonym_scores["Is Synonym"] = synonym_scores["Is Synonym"].astype(bool)

    original_label = synonym_scores[synonym_scores["Is Synonym"] == False]
    synonyms = synonym_scores[synonym_scores["Is Synonym"] == True]
    plot_df = pd.merge(
        original_label,
        synonyms,
        on=["FT-Dataset", "Metric"],
        suffixes=("_original", "_synonym"),
    )

    plot_df["diff"] = plot_df["mean_synonym"] - plot_df["mean_original"]
    plot_df["diff"] = plot_df["diff"].apply(lambda x: x * 100).round(1)
    plot_df["Metric"] = plot_df["Metric"].map(
        {"f_score": "F1", "precision": "Precision", "recall": "Recall"}
    )
    plot_df["FT-Dataset"] = plot_df["FT-Dataset"].apply(lambda x: display_train[x])
    plot_df.drop(columns=["Is Synonym_synonym"], inplace=True)
    plot_df.rename(
        columns={
            "Is Synonym_original": "Is Synonym",
            "diff": "Absolute Diff. Original Label vs. Synonym",
        },
        inplace=True,
    )

    sns.set_theme(style="whitegrid", font_scale=1.8)
    g = sns.catplot(
        data=plot_df,
        kind="bar",
        x="Absolute Diff. Original Label vs. Synonym",
        y="FT-Dataset",
        row="Metric",
        hue="FT-Dataset",
        order=list(display_train.values()),
        hue_order=list(display_train.values()),
        height=3.5,
        aspect=2.2,
        orient="h",
        legend="brief",
    )

    for ax in g.axes.flat:
        ax.get_yaxis().set_visible(False)

    sns.move_legend(
        g,
        "lower center",
        bbox_to_anchor=(0.35, -0.13),
        ncol=2,
        title=None,
        frameon=False,
    )

    g.tight_layout()
    g.savefig(
        f"synonyms_bar.png",
        bbox_inches="tight",
    )
    plt.clf()


if __name__ == "__main__":
    results = pd.read_pickle("/vol/tmp/goldejon/gliner/eval_synonyms/results.pkl")
    plot_synonym_scores(results)
