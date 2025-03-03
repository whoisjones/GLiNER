import json
import pandas as pd

import copy
import torch
import random
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

from typing import List, Dict

from gliner.modules.run_evaluation import create_dataset
import seaborn as sns
import matplotlib.pyplot as plt
from analysis.data import display_eval, display_train
import numpy as np


def plot_distribution_splits():
    df = pd.read_pickle("/vol/tmp/goldejon/gliner/new_splits/splits.pkl")
    for metric in ["entropy", "max"]:
        # Filter the DataFrame for the current metric
        df_long = df[["entity", "eval_dataset", "train_dataset", metric]].copy()
        df_long = df_long.rename(columns={metric: "value"})
        df_long["train_dataset"] = df_long["train_dataset"].replace(display_train)
        df_long["eval_dataset"] = df_long["eval_dataset"].replace(display_eval)
        df_long["metric"] = metric
        metric_display = (
            "Entropy Between Train Label and Eval Label Set"
            if metric == "entropy"
            else "Max. Similarity Between Train Label and Eval Label Set"
        )

        # Calculate the 5th and 95th percentiles for each combination of train and eval datasets
        thresholds = {}
        for train_ds in df_long["train_dataset"].unique():
            quantiles = (
                [
                    0.01 if train_ds == "PileNER" else 0.005,
                    0.95 if train_ds == "PileNER" else 0.995,
                ]
                if metric == "entropy"
                else [
                    0.05 if train_ds == "PileNER" else 0.005,
                    0.99 if train_ds == "PileNER" else 0.995,
                ]
            )
            thresholds[train_ds] = (
                df_long.groupby(["train_dataset", "eval_dataset"])["value"]
                .quantile(quantiles)
                .unstack()
            )

        # Define a function to add vertical lines for the thresholds
        def add_threshold_lines(data, **kwargs):
            train = data["train_dataset"].iloc[0]
            eval_ = data["eval_dataset"].iloc[0]
            low, high = thresholds[train].columns.values.tolist()
            lower_bar = thresholds[train].loc[(train, eval_), low]
            upper_bar = thresholds[train].loc[(train, eval_), high]
            medium_bar = data["value"].quantile([0.5]).iloc[0]

            if metric == "entropy":
                low_label = "Low Label Shift (Low 1% PileNER | Low 0.5% NuNER)"
                high_label = "High Label Shift (Top 5% PileNER | Top 0.5% NuNER)"
                low_color = "tab:blue"
                high_color = "tab:red"
            elif metric == "max":
                low_label = "High Label Shift (Low 5% PileNER | Low 0.5% NuNER)"
                high_label = "Low Label Shift (Top 1% PileNER | Top 0.5% NuNER)"
                low_color = "tab:red"
                high_color = "tab:green"

            ax = plt.gca()  # Get the current axis
            ax.axvline(
                x=lower_bar, color=low_color, linestyle="--", lw=2.5, label=low_label
            )
            ax.axvspan(data["value"].min(), lower_bar, color=low_color, alpha=0.1)

            ax.axvline(
                x=medium_bar,
                color="tab:blue",
                linestyle="--",
                lw=2.5,
                label="Medium Label Shift (Middle 49.5% - 50.5%)",
            )

            ax.axvline(
                x=upper_bar, color=high_color, linestyle="--", lw=2.5, label=high_label
            )
            ax.axvspan(upper_bar, data["value"].max(), color=high_color, alpha=0.1)

            handles, labels = ax.get_legend_handles_labels()
            if handles and labels:
                ax.legend(
                    handles=handles,
                    labels=labels,
                    loc="upper right",
                    fontsize="large",
                    title="Thresholds",
                    title_fontsize="medium",
                    frameon=True,
                )

        # Create the displot for the current metric with adjusted aesthetics
        g = sns.displot(
            data=df_long,
            x="value",
            row="eval_dataset",
            col="train_dataset",
            kind="kde",
            alpha=0.2,
            linewidth=1.5,
            color="tab:orange",
            fill=True,
            facet_kws={"sharey": False, "sharex": False, "margin_titles": True},
            height=4,  # Adjust height for better readability
            aspect=1.8,  # Adjust aspect ratio for better width
        )

        # Iterate through each axis to add borders and a grid
        for ax in g.axes.flat:
            ax.grid(True, linestyle=":", linewidth=0.6)  # Add a light grid
            sns.despine(
                ax=ax, left=False, bottom=False
            )  # Ensure borders on all sides for each subplot
            for spine in ax.spines.values():
                spine.set_visible(True)  # Make sure all borders are visible
                spine.set_linewidth(1.5)  # Make borders slightly thicker

        # Customize the appearance of titles and labels
        g.set_titles(row_template="{row_name}", col_template="{col_name}", size=16)
        g.fig.text(0.5, -0.01, metric_display, ha="center", va="center", fontsize=16)
        g.fig.text(
            -0.01,
            0.5,
            "Density",
            ha="center",
            va="center",
            rotation="vertical",
            fontsize=16,
        )
        for ax in g.axes.flat:
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(
                axis="x", labelsize=13
            )  # Increase font size of x-axis tick labels
            ax.tick_params(
                axis="y", labelsize=13
            )  # Increase font size of y-axis tick labels
        g.fig.subplots_adjust(top=0.9)  # Adjust space at the top for the titles

        # Add the threshold lines using the custom function
        g.map_dataframe(add_threshold_lines)

        handles, labels = g.axes.flat[0].get_legend_handles_labels()

        # Add the overall legend below the entire plot
        g.fig.legend(
            handles=handles,
            labels=labels,
            loc="lower center",  # Position the legend below the plot
            bbox_to_anchor=(
                0.5,
                -0.08,
            ),  # Center the legend horizontally below the plot
            ncol=1,  # Number of columns in the legend
            fontsize="14",
            title="Thresholds",
            title_fontsize=16,
            frameon=True,
        )

        plt.savefig(f"{metric}_distribution.png", dpi=300, bbox_inches="tight")
        plt.close()


def create_splits(
    dataset: List[Dict],
    dataset_name: str,
    filter_by: str = "entropy",
    setting: str = "medium",
):
    df = pd.read_pickle("/vol/tmp/goldejon/gliner/new_splits/splits.pkl")
    df = df[(df["train_dataset"] == dataset_name)]

    selected_entity_types = []
    for benchmark_name in df["eval_dataset"].unique():
        _df = df[(df["eval_dataset"] == benchmark_name)].copy()

        if filter_by == "entropy":
            low_threshold = df[filter_by].quantile(
                0.01 if dataset_name == "pilener_train" else 0.005
            )
            high_threshold = df[filter_by].quantile(
                0.95 if dataset_name == "pilener_train" else 0.995
            )
        elif filter_by == "max":
            low_threshold = df[filter_by].quantile(
                0.05 if dataset_name == "pilener_train" else 0.005
            )
            high_threshold = df[filter_by].quantile(
                0.99 if dataset_name == "pilener_train" else 0.995
            )

        medium_lower_threshold = df[filter_by].quantile(0.495)
        medium_upper_threshold = df[filter_by].quantile(0.505)

        # Define conditions and choices for categorization
        conditions = [
            _df[filter_by] <= low_threshold,  # Bottom
            _df[filter_by].between(
                medium_lower_threshold, medium_upper_threshold
            ),  # Middle
            _df[filter_by] >= high_threshold,  # Top
        ]
        choices = (
            ["easy", "medium", "hard"]
            if filter_by == "entropy"
            else ["hard", "medium", "easy"]
        )

        # Use np.select to create the new column based on the conditions
        _df["difficulty"] = np.select(conditions, choices, default="not relevant")

        selected_entity_types.extend(
            _df[_df["difficulty"] == setting]["entity"].tolist()
        )

    new_dataset = []
    for dp in tqdm(dataset):
        matched_entities = [
            x for x in dp["ner"] if x[-1].lower().strip() in selected_entity_types
        ]
        if matched_entities:
            new_np = copy.deepcopy(dp)
            new_np["ner"] = matched_entities
            new_dataset.append(new_np)

    return new_dataset


def load_train_dataset(dataset_name):
    with open("/vol/tmp/goldejon/gliner/train_datasets/" + dataset_name, "r") as f:
        data = json.load(f)
    return data


def main():
    train_datasets = ["pilener_train.json", "nuner_train.json"]
    benchmark_names = [
        "mit-movie",
        "mit-restaurant",
        "CrossNER_AI",
        "CrossNER_literature",
        "CrossNER_music",
        "CrossNER_politics",
        "CrossNER_science",
    ]

    benchmarks = {}
    for benchmark_name in benchmark_names:
        _, _, test_dataset, entity_types = create_dataset(
            "/vol/tmp/goldejon/gliner/eval_datasets/NER/" + benchmark_name
        )
        benchmarks[benchmark_name] = entity_types

    training_datasets = {}
    for train_dataset_name in train_datasets:
        train_dataset = load_train_dataset(train_dataset_name)
        entity_types = set()
        for dp in train_dataset:
            annotations = [x[-1].lower().strip() for x in dp["ner"]]
            entity_types.update(annotations)
        training_datasets[train_dataset_name] = list(entity_types)

    batch_size = 256
    model = SentenceTransformer("all-mpnet-base-v2").to("cuda")
    eval_encodings = {}
    for benchmark_name, entity_types in benchmarks.items():
        embeddings = model.encode(entity_types, convert_to_tensor=True, device="cuda")
        eval_encodings[benchmark_name] = embeddings

    results = {}
    for dataset_name, entity_types in training_datasets.items():
        for i in tqdm(range(0, len(entity_types), batch_size)):
            dataset_name = dataset_name.split(".")[0]
            batch = entity_types[i : i + batch_size]
            embeddings = model.encode(batch, convert_to_tensor=True, device="cuda")
            for benchmark_name, eval_embeddings in eval_encodings.items():
                similarities = torch.clamp(
                    cosine_similarity(
                        embeddings.unsqueeze(1),
                        eval_embeddings.unsqueeze(0),
                        dim=2,
                    ),
                    min=0.0,
                    max=1.0,
                )
                probabilities = torch.nn.functional.softmax(similarities / 0.01, dim=1)
                entropy_values = -torch.sum(
                    probabilities * torch.log(probabilities + 1e-10), dim=1
                )
                max_values, _ = torch.max(similarities, dim=1)

                if dataset_name not in results:
                    results[dataset_name] = {}
                if benchmark_name not in results[dataset_name]:
                    results[dataset_name][benchmark_name] = {}

                for j, entity in enumerate(batch):
                    if entity not in results[dataset_name][benchmark_name]:
                        results[dataset_name][benchmark_name][entity] = {}
                    results[dataset_name][benchmark_name][entity]["entropy"] = (
                        entropy_values[j].cpu().numpy().item()
                    )
                    results[dataset_name][benchmark_name][entity]["max"] = (
                        max_values[j].cpu().numpy().item()
                    )

    entries = []
    for dataset_name, eval_comparisons in results.items():
        for benchmark_name, mapping in eval_comparisons.items():
            for entity, values in mapping.items():
                entries.append(
                    {
                        "entity": entity,
                        "entropy": values["entropy"],
                        "max": values["max"],
                        "eval_dataset": benchmark_name,
                        "train_dataset": dataset_name,
                    }
                )
    df = pd.DataFrame.from_dict(entries, orient="columns")
    df.to_pickle("/vol/tmp/goldejon/gliner/new_splits/splits.pkl")
