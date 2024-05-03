import random

from data import (
    get_train_datasets_stats,
    get_eval_datasets_stats,
    get_eval_scores,
    bin_eval_labels,
    exposure_type_during_training,
    count_label_seen,
    compute_overlaps,
    display_train,
    display_eval,
)

import seaborn as sns


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


def plot_scatter(scores):
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


def main():
    random.seed(42)

    train_statistics = get_train_datasets_stats()
    eval_statistics = get_eval_datasets_stats()
    scores = get_eval_scores()

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

    scores = bin_eval_labels(scores)
    scores = scores.apply(exposure_type_during_training, axis=1)
    scores = scores.apply(count_label_seen, axis=1)

    scores.rename(columns={"f_score": "f1"}, inplace=True)

    plot_scatter(scores)


if __name__ == "__main__":
    main()
