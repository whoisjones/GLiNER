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

from modules.run_evaluation import create_dataset

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


def get_synonyms_scores(
    train_datasets: list,
    eval_datasets: list,
    scores_path: str = "/vol/tmp/goldejon/gliner/eval_synonyms",
):
    synonym_scores = pd.DataFrame()
    for train_dataset in train_datasets:
        for eval_dataset in eval_datasets:
            paths = glob.glob(f"{scores_path}/{train_dataset}/*/{eval_dataset}.pkl")
            for path in paths:
                scores = pd.read_pickle(path)
                scores["original_label"] = scores["original_label"].str.lower()
                scores["synonym_label"] = scores["synonym_label"].str.lower()
                timestamp = path.split("/")[-2]
                scores["Timestamp"] = timestamp
                scores["Dataset (Train)"] = train_dataset
                synonym_scores = pd.concat([synonym_scores, scores])

    synonym_scores.rename(
        columns={
            "dataset_name": "Dataset (Eval)",
            "is_synonym": "Is Synonym",
            "metric": "Metric",
            "value": "Value",
        },
        inplace=True,
    )

    synonym_scores["Dataset (Train)"] = (
        synonym_scores["Dataset (Train)"].str.split("_").str[0].str.capitalize()
    )

    def format_dataset_name(dataset_name):
        if dataset_name.startswith("CrossNER"):
            if "AI" in dataset_name:
                dataset_name = dataset_name.split("_")[1]
            else:
                dataset_name = dataset_name.split("_")[1].capitalize()
        else:
            dataset_name = dataset_name.split("-")[1].capitalize()
        return dataset_name

    synonym_scores["Dataset (Eval)"] = synonym_scores["Dataset (Eval)"].apply(
        format_dataset_name
    )

    synonym_scores.reset_index(drop=True, inplace=True)
    return synonym_scores


def plot_synonym_scores(synonym_scores: pd.DataFrame):
    # Plot Evaluation Dataset Scores
    synonym_scores = (
        synonym_scores[["Is Synonym", "Dataset (Train)", "Metric", "Value"]]
        .groupby(["Is Synonym", "Dataset (Train)", "Metric"])
        .mean()
        .reset_index()
    )

    original_scores = synonym_scores[synonym_scores["Is Synonym"] == False]
    synonym_scores = synonym_scores[synonym_scores["Is Synonym"] == True]
    result = pd.merge(original_scores, synonym_scores, on=["Dataset (Train)", "Metric"])
    result["Diff. in pp. with synonyms vs. original label"] = (
        result["Value_y"] - result["Value_x"]
    )

    sns.set_theme(style="whitegrid", font_scale=1)
    g = sns.catplot(
        data=result,
        kind="bar",
        x="Diff. in pp. with synonyms vs. original label",
        y="Dataset (Train)",
        row="Metric",
        hue="Dataset (Train)",
        order=["Ontonotes", "Fewnerd", "Litset", "Pilener"],
        hue_order=["Ontonotes", "Fewnerd", "Litset", "Pilener"],
        height=1.75,
        aspect=2.5,
        orient="h",
        legend="brief",
    )
    g.set(xlim=(-35, 15))

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
        f"synonyms_results.png",
        bbox_inches="tight",
    )


def latex_table(synonym_scores):
    latex_data = (
        latex_data[["is_synonym", "dataset_name", "Dataset (Train)", "value"]]
        .groupby(["is_synonym", "dataset_name", "Dataset (Train)"])
        .mean()
        .reset_index()
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--output_path", type=str, default="/vol/tmp/goldejon/gliner/eval_synonyms"
    )
    args = parser.parse_args()

    train_datasets_paths = [
        "ontonotes",
        "fewnerd",
        "litset",
        "pilener_train",
    ]
    eval_datasets_paths = [
        "mit-movie",
        "mit-restaurant",
        "CrossNER_AI",
        "CrossNER_literature",
        "CrossNER_music",
        "CrossNER_politics",
        "CrossNER_science",
    ]

    logger.info("Get Synonym Scores.")
    synonym_scores = get_synonyms_scores(train_datasets_paths, eval_datasets_paths)

    plot_synonym_scores(synonym_scores)
    # latex_table(synonym_scores)
