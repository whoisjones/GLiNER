import numpy as np
import seaborn as sns

from data import (
    get_train_datasets_stats,
    get_eval_datasets_stats,
    compute_overlaps,
    display_train,
    display_eval,
)


def plot_overlaps_percentage(overlap_information):
    # Plot Overlapping Dataset Statistics
    overlaps_binary = overlap_information[
        [
            "train_dataset",
            "eval_dataset",
            "exact_overlap_percentage",
            "partial_overlap_percentage",
        ]
    ].rename(
        columns={
            "exact_overlap_percentage": "Exact Overlap",
            "partial_overlap_percentage": "Partial Overlap",
        }
    )

    overlaps_binary = overlaps_binary.melt(
        id_vars=["train_dataset", "eval_dataset"],
        var_name="Exposure Type",
        value_name="Overlap (in %)",
    )

    overlaps_binary["train_dataset"] = overlaps_binary["train_dataset"].apply(
        lambda x: display_train.get(x)
    )

    overlaps_binary["eval_dataset"] = overlaps_binary["eval_dataset"].apply(
        lambda x: display_eval.get(x)
    )

    overlaps_binary.rename(
        columns={"train_dataset": "FT-Dataset", "eval_dataset": "Eval Benchmark"},
        inplace=True,
    )

    sns.set_theme(style="whitegrid")
    g = sns.FacetGrid(
        overlaps_binary,
        col="FT-Dataset",
        height=2.6,
        aspect=1.7,
        col_wrap=3,
        col_order=list(display_train.values()),
    )

    g.map_dataframe(
        sns.barplot,
        "Eval Benchmark",
        "Overlap (in %)",
        hue="Exposure Type",
        palette="deep",
    )
    g.add_legend()

    g.set(yticks=np.arange(0, 1.01, 0.2))
    y_labels = [int(x) for x in g.axes.flat[0].get_yticks() * 100]
    g.set_yticklabels(y_labels)
    xlabels = g.axes.flat[-1].get_xticklabels()
    g.set_xticklabels(xlabels, rotation=90)
    g.set_axis_labels("", "Overlap (in %)")

    sns.move_legend(
        g,
        "upper center",
        bbox_to_anchor=(0.45, 1.05),
        ncol=2,
        frameon=False,
    )

    g.tight_layout()
    g.savefig(
        f"overlap_between_datasets_percentage.png",
        bbox_inches="tight",
    )


def plot_overlaps_count(overlap_information):
    # Plot Overlapping Dataset Statistics
    overlaps_count = overlap_information[
        [
            "train_dataset",
            "eval_dataset",
            "exact_overlap_sum",
            "partial_overlap_sum",
        ]
    ].rename(
        columns={
            "exact_overlap_sum": "Exact Overlap",
            "partial_overlap_sum": "Partial Overlap",
        }
    )

    overlaps_count["Partial Overlap"] = (
        overlaps_count["Partial Overlap"] - overlaps_count["Exact Overlap"]
    )

    overlaps_count = overlaps_count.melt(
        id_vars=["train_dataset", "eval_dataset"],
        var_name="Exposure Type",
        value_name="# Entities seen in Pretraining",
    )

    overlaps_count["train_dataset"] = overlaps_count["train_dataset"].apply(
        lambda x: display_train.get(x)
    )

    overlaps_count["eval_dataset"] = overlaps_count["eval_dataset"].apply(
        lambda x: display_eval.get(x)
    )

    overlaps_count.rename(
        columns={"train_dataset": "FT-Dataset", "eval_dataset": "Eval Benchmark"},
        inplace=True,
    )

    sns.set_theme(style="whitegrid")
    g = sns.FacetGrid(
        overlaps_count,
        col="FT-Dataset",
        col_order=list(display_train.values()),
        height=2.6,
        aspect=1.7,
        col_wrap=3,
    )

    g.map_dataframe(
        sns.barplot,
        "Eval Benchmark",
        "# Entities seen in Pretraining",
        hue="Exposure Type",
        palette="deep",
    )

    for ax in g.axes.flat:
        ax.set_yscale("symlog")
        ax.set_ylim(0)

    g.add_legend()

    xlabels = g.axes.flat[-1].get_xticklabels()
    g.set_xticklabels(xlabels, rotation=90)
    g.set_axis_labels("", "# Entities Overlapping")

    sns.move_legend(
        g,
        "upper center",
        bbox_to_anchor=(0.45, 1.05),
        ncol=2,
        frameon=False,
    )

    g.tight_layout()
    g.savefig(
        f"overlap_between_datasets_count.png",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    train_statistics = get_train_datasets_stats()
    eval_statistics = get_eval_datasets_stats()

    overlap_information = train_statistics.merge(
        eval_statistics,
        how="cross",
    )
    overlap_information = overlap_information.apply(compute_overlaps, axis=1)

    plot_overlaps_percentage(overlap_information)
    plot_overlaps_count(overlap_information)
