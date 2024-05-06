import sys

sys.path.append("/vol/fob-vol7/mi18/goldejon/GLiNER")

import pandas as pd
import seaborn as sns

from evaluate_hierarchy import hierarchy
from data import display_train


def plot_hierarchy(scores):
    inverse_hierarchy = {key: v for v, keys in hierarchy.items() for key in keys}

    scores["top_level_entity"] = scores["entity"].apply(
        lambda entity: (
            inverse_hierarchy[entity] if entity in inverse_hierarchy else entity
        )
    )

    scores["is_top_level"] = scores["entity"].apply(
        lambda entity: entity in hierarchy.keys()
    )

    scores = scores.groupby(["top_level_entity", "is_top_level", "train_dataset"]).agg(
        {
            "f_score": ["mean", "std"],
            "precision": ["mean", "std"],
            "recall": ["mean", "std"],
        }
    )

    diff_df = pd.DataFrame()
    for top_level_entity in hierarchy.keys():
        diff = (
            (
                scores.loc[(top_level_entity, False)]
                - scores.loc[(top_level_entity, True)]
            )
            * 100
        ).round(2)

        diff["top_level_entity"] = top_level_entity
        diff_df = pd.concat([diff_df, diff])

    metric_display = {
        "f_score": "F1",
        "precision": "Precision",
        "recall": "Recall",
    }

    for metric in ["f_score", "precision", "recall"]:
        plot_df = diff_df[[metric, "top_level_entity"]].reset_index()
        plot_df.columns = [
            "FT-Dataset",
            f"Absolute Diff. Top-Level vs. Subclass Entity ({metric_display[metric]})",
            "Std",
            "Entity",
        ]

        sns.set_theme(style="whitegrid", font_scale=1)
        g = sns.catplot(
            data=plot_df,
            kind="bar",
            x=f"Absolute Diff. Top-Level vs. Subclass Entity ({metric_display[metric]})",
            y="FT-Dataset",
            row="Entity",
            hue="FT-Dataset",
            order=list(display_train.values()),
            hue_order=list(display_train.values()),
            height=1.75,
            aspect=2.5,
            orient="h",
            legend="brief",
        )
        g.set_xlabels(
            f"Absolute Diff. Top-Level vs. Subclass Entity ({metric_display[metric]})",
            fontsize=11,
        )

        for ax in g.axes.flat:
            ax.get_yaxis().set_visible(False)

        sns.move_legend(
            g,
            "lower center",
            bbox_to_anchor=(0.35, -0.11),
            ncol=2,
            title=None,
            frameon=False,
        )

        g.tight_layout()
        g.savefig(
            f"hierarchy_{metric}.png",
            bbox_inches="tight",
        )


if __name__ == "__main__":
    results = pd.read_pickle("/vol/tmp/goldejon/gliner/eval_hierarchy/results.pkl")
    results = results[results["entity"] != "average"]
    results["train_dataset"] = results["train_dataset"].apply(
        lambda x: display_train[x]
    )

    plot_hierarchy(results)
