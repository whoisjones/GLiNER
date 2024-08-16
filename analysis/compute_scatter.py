import random

import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from data import (
    compute_overlaps,
    count_label_seen,
    display_eval,
    display_train,
    exposure_type_during_training,
    get_eval_datasets_stats,
    get_eval_scores,
    get_train_datasets_stats,
)


def plot_exact_scatter(scores):
    for metric in ["precision", "recall", "f1"]:
        sns.set_theme(style="whitegrid")
        scores["Label Seen during Pretraining"] = scores[
            "times_entity_seen_exact"
        ].apply(lambda x: "exact" if x > 0 else "not seen")

        g = sns.FacetGrid(
            scores,
            col="FT-Dataset",
            col_order=list(display_train.values()),
            col_wrap=3,
            height=3.5,
            aspect=1.2,
        )

        g.map_dataframe(
            sns.scatterplot,
            "times_entity_seen_exact",
            metric,
            size="# Evaluation Labels",
            hue="Label Seen during Pretraining",
            sizes={
                "0-50": 50,
                "50-100": 100,
                "100-250": 150,
                "250-500": 200,
                "500+": 250,
            },
            palette="deep",
            hue_order=["exact", "not seen"],
            size_order=["0-50", "50-100", "100-250", "250-500", "500+"],
            alpha=0.7,
        )

        for ax in g.axes.flat:
            ax.set_xscale("symlog")
            ax.set_xlim(-0.5)

        g.set_axis_labels("", f"{metric.capitalize()}")
        g.add_legend()
        sns.move_legend(
            g, "lower center", bbox_to_anchor=(0.43, -0.13), ncol=3, frameon=False
        )
        g.fig.text(
            0.43,
            0.0,
            "Number of Evaluation Entities seen during Pretraining",
            ha="center",
        )

        g.tight_layout()
        g.savefig(
            f"scores_by_labels_and_overlap_exact ({metric}).png",
            bbox_inches="tight",
        )


def plot_partial_scatter(scores):
    for metric in ["precision", "recall", "f1"]:
        sns.set_theme(style="whitegrid")
        g = sns.FacetGrid(
            scores,
            col="FT-Dataset",
            col_order=list(display_train.values()),
            col_wrap=3,
            height=3.5,
            aspect=1.2,
        )

        g.map_dataframe(
            sns.scatterplot,
            "times_entity_seen_partial",
            metric,
            size="# Evaluation Labels",
            hue="Label Seen during Pretraining",
            sizes={
                "0-50": 50,
                "50-100": 100,
                "100-250": 150,
                "250-500": 200,
                "500+": 250,
            },
            palette="deep",
            hue_order=["exact + partial", "exact only", "partial only", "not seen"],
            size_order=["0-50", "50-100", "100-250", "250-500", "500+"],
            alpha=0.7,
        )

        for ax in g.axes.flat:
            ax.set_xscale("symlog")
            ax.set_xlim(-0.5)

        g.set_axis_labels("", f"{metric.capitalize()}")
        handles, labels = g.axes[-1].get_legend_handles_labels()
        handles.insert(labels.index("# Evaluation Labels"), handles[0])
        labels.insert(labels.index("# Evaluation Labels"), "")
        legend_data = {label: handle for handle, label in zip(handles, labels)}

        g.add_legend(legend_data)
        sns.move_legend(
            g, "lower center", bbox_to_anchor=(0.43, -0.13), ncol=4, frameon=False
        )
        g.fig.text(
            0.43,
            0.0,
            "Number of Evaluation Entities seen during Pretraining",
            ha="center",
        )

        g.tight_layout()
        g.savefig(
            f"scores_by_labels_and_overlap_partial ({metric}).png",
            bbox_inches="tight",
        )


def plot_old_scatter(scores):
    scores = scores.rename(
        columns={
            "exposure_type": "Label Seen during Pretraining",
            "eval_label_bin": "# Evaluation Labels",
        }
    )

    scores["train_dataset"] = scores["train_dataset"].apply(
        lambda x: display_train.get(x)
    )

    scores["eval_dataset"] = scores["eval_dataset"].apply(lambda x: display_eval.get(x))

    scores.rename(
        columns={"train_dataset": "FT-Dataset", "eval_dataset": "Eval Benchmark"},
        inplace=True,
    )

    plot_exact_scatter(scores)
    plot_partial_scatter(scores)


def plot_scatter(scores):
    agg_columns = [
        "entity",
        "eval_dataset",
        "train_dataset",
        "exposure_type",
        "times_entity_seen_exact",
        "times_entity_seen_partial",
    ]
    filter_columns = [
        "precision",
        "recall",
        "f1",
        "entity",
        "eval_dataset",
        "train_dataset",
        "exposure_type",
        "times_entity_seen_exact",
        "times_entity_seen_partial",
    ]
    scores = scores[filter_columns].groupby(agg_columns).mean().reset_index()
    filtered_display_train = {
        key: value
        for key, value in display_train.items()
        if key in scores["train_dataset"].values
    }

    for metric in ["precision", "recall", "f1"]:
        fig, axes = plt.subplots(1, 5, figsize=(12, 3), sharey=True)

        for i, (train_dataset, train_dataset_display) in enumerate(
            filtered_display_train.items()
        ):

            dataset_scores = scores[scores["train_dataset"] == train_dataset]

            zeroshot_scores = dataset_scores[
                dataset_scores["exposure_type"] == "true zero-shot"
            ]
            overlap_scores = dataset_scores[
                dataset_scores["exposure_type"] != "true zero-shot"
            ]
            overlap_color = "mediumslateblue"
            no_overlap_color = "limegreen"

            g = sns.regplot(
                x="times_entity_seen_partial",
                y=metric,
                data=overlap_scores,
                color=overlap_color,
                ax=axes[i],
                logx=True,
                scatter_kws={"edgecolors": "none", "s": 4},
                line_kws={"color": overlap_color},
            )

            g.axhline(
                zeroshot_scores[metric].mean(),
                color=no_overlap_color,
                linestyle="--",
                label=f"{zeroshot_scores[metric].mean():.2f}",
            )

            g.set_title(f"(fine-tuned on:) {train_dataset_display}", fontsize=10)
            g.set_xscale("log")
            g.set_xlabel("")
            g.minorticks_off()
            g.set_ylim(0, 1)

            if i == 0:
                g.set_ylabel(f"{metric.capitalize()}")
            else:
                g.set_ylabel("")
            if i == 2:
                g.set_xlabel("Frequency of Evaluation Label in Fine-Tuning Dataset")

            slope, intercept, r, p, sterr = scipy.stats.linregress(
                x=np.log10(g.get_lines()[0].get_xdata()),
                y=g.get_lines()[0].get_ydata(),
            )
            x_label = rf"${slope:.2f}⋅\log_{{10}}(x) + {intercept:.2f}$"
            g.lines[0].set_label(x_label)

            g.legend(prop={"size": 8}, loc="upper left")

        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Scatter",
                markerfacecolor=overlap_color,
                markersize=8,
            ),
            Patch(facecolor=overlap_color, edgecolor="none", label="Overlap"),
            Patch(
                facecolor=no_overlap_color,
                edgecolor="none",
                label="No overlap (true zero-shot)",
            ),
        ]
        fig.legend(
            handles,
            [
                "Unique Evaluation Label",
                "Overlaps w/ Fine-Tuning Labels",
                "True Zero-Shot Transfer",
            ],
            loc="lower center",
            ncols=3,
            bbox_to_anchor=(0.5, -0.075),
        )
        fig.suptitle(
            "Transfer Performance Linked to Prevalence and Extend of Label Overlaps between Evaluation and Fine-Tuning Datasets",
            y=0.95,
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(f"{metric}_correlation.png", dpi=1000, bbox_inches="tight")
        plt.close(fig)


def main():
    random.seed(42)

    base_path = "/home/ec2-user/paper_data"
    # base_path = "/vol/tmp/goldejon/gliner"
    train_statistics = get_train_datasets_stats(base_path=base_path)
    train_statistics = train_statistics[train_statistics["train_dataset"] != "fewnerd"]
    train_statistics = train_statistics[
        train_statistics["train_dataset"] != "ontonotes"
    ]
    eval_statistics = get_eval_datasets_stats(base_path=base_path)
    scores = get_eval_scores(base_path=base_path)

    overlaps = train_statistics.merge(
        eval_statistics,
        how="cross",
    )
    overlaps = overlaps.apply(compute_overlaps, axis=1)

    scores = scores.merge(
        overlaps[
            [
                "train_dataset",
                "eval_dataset",
                "eval_labels_counter",
                "exact_overlap",
                "partial_overlap",
                "exact_overlap_counter",
                "partial_overlap_counter",
            ]
        ],
        how="inner",
        on=["train_dataset", "eval_dataset"],
    )

    scores = scores.apply(exposure_type_during_training, axis=1)
    scores = scores.apply(count_label_seen, axis=1)

    scores.rename(columns={"f_score": "f1"}, inplace=True)

    # plot_old_scatter(scores)
    plot_scatter(scores)


if __name__ == "__main__":
    main()
