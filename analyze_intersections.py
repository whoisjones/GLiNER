import os
import json
import glob
import random
import logging
from argparse import ArgumentParser
from collections import Counter

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from evaluate_intersections import labels

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create formatter with timestamp
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Create console handler and set level to debug
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Add formatter to console handler
console_handler.setFormatter(formatter)

# Add console handler to logger
logger.addHandler(console_handler)


def get_intersection_scores(
    train_datasets: list,
    scores_path: str = "/vol/tmp/goldejon/gliner/eval_intersections",
):
    intersection_scores = pd.DataFrame()
    for train_dataset in train_datasets:
        for coarse_label, _ in labels.items():
            for is_coarse in ["coarse", "fine"]:
                paths = glob.glob(
                    f"{scores_path}/{train_dataset}/*/{coarse_label + '_' + is_coarse}.pkl"
                )
                for path in paths:
                    scores = pd.read_pickle(path)
                    scores["Dataset"] = train_dataset
                    scores["Granularity"] = (
                        "Top Type" if is_coarse == "coarse" else "Intersections"
                    )
                    scores["Label"] = coarse_label
                    scores = pd.melt(
                        scores.reset_index().rename(columns={"index": "Metric"}),
                        id_vars=[
                            "Metric",
                            "Dataset",
                            "Granularity",
                            "Label",
                        ],
                        var_name="Intersection Label",
                        value_name="Value",
                    )
                    intersection_scores = pd.concat([intersection_scores, scores])

    intersection_scores["Dataset"] = (
        intersection_scores["Dataset"].str.split("_").str[0].str.capitalize()
    )

    intersection_scores_with_diff = pd.DataFrame()
    for ds in intersection_scores["Dataset"].unique():
        for metric in intersection_scores["Metric"].unique():
            for label in intersection_scores["Label"].unique():
                df_for_ds = intersection_scores[
                    (intersection_scores["Dataset"] == ds)
                    & (intersection_scores["Label"] == label)
                    & (intersection_scores["Metric"] == metric)
                    & (intersection_scores["Intersection Label"] == "overall")
                ]

                f1_top = df_for_ds[(df_for_ds["Granularity"] == "Top Type")][
                    "Value"
                ].values[0]

                f1_intersect = df_for_ds[(df_for_ds["Granularity"] == "Intersections")][
                    "Value"
                ].values[0]

                intersection_scores_with_diff = pd.concat(
                    [
                        intersection_scores_with_diff,
                        df_for_ds,
                        pd.DataFrame.from_dict(
                            {
                                "Metric": [metric],
                                "Dataset": [ds],
                                "Label": [label],
                                "Granularity": ["Diff"],
                                "Intersection Label": ["overall"],
                                "Value": [f1_intersect - f1_top],
                            },
                        ),
                    ]
                )

    intersection_scores_with_diff["Value"] = intersection_scores_with_diff[
        "Value"
    ].apply(lambda x: round(x * 100, 2))

    latex_scores = pd.DataFrame()
    for dataset in intersection_scores_with_diff["Dataset"].unique():
        scores = intersection_scores_with_diff[
            (intersection_scores_with_diff["Intersection Label"] == "overall")
            & (intersection_scores_with_diff["Dataset"] == dataset)
            & (intersection_scores_with_diff["Metric"] == "f1")
        ]

        scores = scores[["Dataset", "Label", "Granularity", "Metric", "Value"]]

        scores = scores.pivot_table(
            index=["Dataset", "Label", "Granularity"],
            columns="Metric",
            values="Value",
        )

        latex_scores = pd.concat([latex_scores, scores])

    sort_order = {}
    i = 0
    for ds in latex_scores.index.get_level_values("Dataset").unique():
        for label in latex_scores.index.get_level_values("Label").unique():
            for granularity in ["Intersections", "Top Type", "Diff"]:
                sort_order[(ds, label, granularity)] = i
                i += 1

    latex_scores["sortby"] = latex_scores.index.map(sort_order)
    latex_scores.sort_values("sortby", inplace=True)
    del latex_scores["sortby"]

    print(latex_scores.to_latex(float_format="{:0.2f}".format, multirow=True))

    latex_scores.reset_index(inplace=True)
    latex_scores = latex_scores[latex_scores["Granularity"] == "Diff"]
    latex_scores.rename(
        columns={"f1": "Diff. in pp. intersection vs. top-level label(s)"}, inplace=True
    )
    sns.set_theme(style="whitegrid", font_scale=1)
    g = sns.catplot(
        data=latex_scores,
        kind="bar",
        x="Diff. in pp. intersection vs. top-level label(s)",
        y="Dataset",
        row="Label",
        hue="Dataset",
        order=["Ontonotes", "Fewnerd", "Litset", "Pilener"],
        hue_order=["Ontonotes", "Fewnerd", "Litset", "Pilener"],
        height=1.75,
        aspect=2.5,
        orient="h",
        legend="brief",
    )
    g.set(xlim=(-35, 25))

    for ax in g.axes.flat:
        ax.get_yaxis().set_visible(False)

    sns.move_legend(
        g,
        "lower center",
        bbox_to_anchor=(0.35, -0.08),
        ncol=2,
        title=None,
        frameon=False,
    )

    g.tight_layout()
    g.savefig(
        f"intersection_results.png",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--output_path", type=str, default="/vol/tmp/goldejon/gliner/eval_intersections"
    )
    args = parser.parse_args()

    train_datasets_paths = [
        "ontonotes",
        "fewnerd",
        "litset",
        "pilener_train",
    ]

    logger.info("Get Intersection Scores.")
    synonym_scores = get_intersection_scores(train_datasets_paths)

    # plot_synonym_scores(synonym_scores)
    # latex_table(synonym_scores)
