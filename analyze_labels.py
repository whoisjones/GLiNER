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


def get_train_labels(
    train_dataset: str,
    train_datasets_path: str = "/vol/tmp/goldejon/gliner/train_datasets",
):
    train_steps = 30000
    batch_size = 8
    required_examples = train_steps * batch_size

    if train_dataset == "litset":
        with open("/vol/tmp/goldejon/ner4all/loner/labelID2label.json", "r") as f:
            id2label = json.load(f)
        id2label.pop("0")

        repeats = required_examples // len(id2label)
        remains = required_examples - (repeats * len(id2label))

        data = []
        if repeats > 0:
            data = [label for label in id2label.values()] * repeats
        if remains > 0:
            data = data + random.sample([label for label in id2label.values()], remains)

        labels = []
        for label in data:
            label_type = random.choice(["description", "labels"])
            fallback_type = "description" if label_type == "labels" else "labels"
            if label_type in label:
                labels.append(
                    random.choice(label[label_type])
                    if label_type == "labels"
                    else label[label_type]
                )
            elif fallback_type in label:
                labels.append(
                    random.choice(label[fallback_type])
                    if fallback_type == "labels"
                    else label[fallback_type]
                )
            else:
                labels.append("miscellaneous")
    else:
        with open(os.path.join(train_datasets_path, f"{train_dataset}.json"), "r") as f:
            data = json.load(f)

        repeats = required_examples // len(data)
        remains = required_examples - (repeats * len(data))

        if repeats > 0:
            data = data * repeats
        if remains > 0:
            data = data + random.sample(data, remains)

        labels = []
        for dp in data:
            for entity in dp["ner"]:
                labels.append(entity[-1])
        labels = labels

    return labels


def get_train_datasets_stats(train_datasets: list):
    train_datasets_stats = pd.DataFrame()
    for train_dataset in train_datasets:
        train_labels = get_train_labels(train_dataset)
        train_labels = [label.lower() for label in train_labels]

        train_labels_binary = set(train_labels)
        train_labels_count = Counter(train_labels)

        df = pd.DataFrame(
            {
                "Dataset": [train_dataset],
                "Labels": [train_labels_binary],
                "Labels (# Entities)": [train_labels_count],
            }
        )

        train_datasets_stats = pd.concat([train_datasets_stats, df])

    train_datasets_stats.reset_index(drop=True, inplace=True)

    return train_datasets_stats


def get_eval_datasets_stats(
    eval_datasets_paths: list,
    eval_datasets_base_path: str = "/vol/tmp/goldejon/gliner/eval_datasets/NER",
):
    eval_datasets_stats = pd.DataFrame()
    for eval_dataset_path in eval_datasets_paths:
        _, _, eval_dataset, _ = create_dataset(
            os.path.join(eval_datasets_base_path, eval_dataset_path)
        )

        eval_labels = []
        for dp in eval_dataset:
            for entity in dp["ner"]:
                eval_labels.append(entity[-1].lower())

        eval_labels_binary = set(eval_labels)
        eval_labels_count = Counter(eval_labels)

        df = pd.DataFrame(
            {
                "Dataset": [eval_dataset_path],
                "Labels": [eval_labels_binary],
                "Labels (# Entities)": [eval_labels_count],
            }
        )

        eval_datasets_stats = pd.concat([eval_datasets_stats, df])

    eval_datasets_stats.reset_index(drop=True, inplace=True)

    return eval_datasets_stats


def get_eval_datasets_scores(
    eval_datasets: list,
    train_datasets: list,
    scores_path: str = "/vol/tmp/goldejon/gliner/eval",
):
    eval_datasets_scores = pd.DataFrame()
    for train_dataset in train_datasets:
        for eval_dataset in eval_datasets:
            paths = glob.glob(f"{scores_path}/{train_dataset}/*/{eval_dataset}.pkl")
            for path in paths:
                scores = pd.read_pickle(path)
                scores["entity"] = scores["entity"].str.lower()
                timestamp = path.split("/")[-2]
                scores["Timestamp"] = timestamp
                scores["Dataset (Train)"] = train_dataset
                eval_datasets_scores = pd.concat([eval_datasets_scores, scores])

    eval_datasets_scores.reset_index(drop=True, inplace=True)
    eval_datasets_scores.rename(
        columns={
            "metric": "Metric",
            "entity": "Label",
            "value": "Value",
            "dataset_name": "Dataset (Eval)",
            "zeroshot": "Zeroshot",
        },
        inplace=True,
    )
    return eval_datasets_scores


def compute_overlaps(row):
    exact_binary_counter = 0
    exact_example_sum = 0
    exact_example_counter = Counter()

    exact_matches = set(
        [tl for tl in row["Labels (Train)"] if tl in row["Labels (Eval)"]]
    )
    for exact_match in exact_matches:
        exact_binary_counter += 1
        num_examples_seen_for_label = row["Labels (# Entities) (Train)"][exact_match]
        exact_example_sum += num_examples_seen_for_label
        exact_example_counter.update({exact_match: num_examples_seen_for_label})

    substring_example_sum = 0
    substring_example_counter = Counter()
    substring_matches = set()
    for el in row["Labels (Eval)"]:
        for tl in row["Labels (Train)"]:
            if el in tl:
                num_examples_seen_for_label = row["Labels (# Entities) (Train)"][tl]
                substring_matches.add(el)
                substring_example_sum += num_examples_seen_for_label
                substring_example_counter.update({el: num_examples_seen_for_label})

    exact_overlap = exact_binary_counter / len(row["Labels (Eval)"])
    partial_overlap = len(substring_matches) / len(row["Labels (Eval)"])

    row["Exact Overlaps (Labels)"] = exact_matches
    row["Partial Overlaps (Labels)"] = substring_matches
    row["Exact Overlap (in %)"] = exact_overlap
    row["Partial Overlap (in %)"] = partial_overlap
    row["Exact Overlap (# Total Entities)"] = exact_example_sum
    row["Partial Overlap (# Total Entities)"] = substring_example_sum
    row["Exact Overlap (# Entities per Label)"] = exact_example_counter
    row["Partial Overlap (# Entities per Label)"] = substring_example_counter

    return row


def exposure_type_during_training(row):
    exact_seen = row["Exact Overlap (# Entities per Label)"][row["Label"]]
    exact_and_partially_seen = row["Partial Overlap (# Entities per Label)"][
        row["Label"]
    ]
    partially_seen = exact_and_partially_seen - exact_seen
    if exact_seen and partially_seen:
        seen_during_training = "exact + partial"
    elif exact_seen:
        seen_during_training = "exact only"
    elif partially_seen:
        seen_during_training = "partial only"
    else:
        seen_during_training = "not seen"

    row["Exposure Type during Training"] = seen_during_training

    return row


def count_label_seen(row):
    row["Label Seen (Exact)"] = row["Exact Overlap (# Entities per Label)"].get(
        row["Label"], 0
    )
    row["Label Seen (Partial)"] = row["Partial Overlap (# Entities per Label)"].get(
        row["Label"], 0
    )
    return row


def flatten_num_target_labels(row):
    row["Count Evaluation Labels"] = row["Labels (# Entities) (Eval)"].get(
        row["Label"], 0
    )
    return row


def plot_overlaps_percentage(overlap_information):
    # Plot Overlapping Dataset Statistics
    overlaps_binary = overlap_information[
        [
            "Dataset (Train)",
            "Dataset (Eval)",
            "Exact Overlap (in %)",
            "Partial Overlap (in %)",
        ]
    ].rename(
        columns={
            "Exact Overlap (in %)": "Exact Overlap",
            "Partial Overlap (in %)": "Partial Overlap",
        }
    )

    overlaps_binary = overlaps_binary.melt(
        id_vars=["Dataset (Train)", "Dataset (Eval)"],
        var_name="Exposure Type",
        value_name="Overlap (in %)",
    )

    overlaps_binary["Dataset (Train)"] = (
        overlaps_binary["Dataset (Train)"].str.split("_").str[0].str.capitalize()
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

    overlaps_binary["Dataset (Eval)"] = overlaps_binary["Dataset (Eval)"].apply(
        format_dataset_name
    )

    overlaps_binary = overlaps_binary.rename(
        columns={"Dataset (Train)": "Pretraining Dataset"}
    )

    sns.set_theme(style="whitegrid")
    g = sns.FacetGrid(
        overlaps_binary,
        col="Pretraining Dataset",
        height=4,
        aspect=1.2,
    )

    g.map_dataframe(
        sns.barplot,
        "Dataset (Eval)",
        "Overlap (in %)",
        hue="Exposure Type",
        palette="deep",
    )
    g.add_legend()

    g.set_xticklabels(rotation=30)
    g.set_axis_labels("", "Overlap (in %)")
    g.set(yticks=np.arange(0, 1.01, 0.2))
    y_labels = [int(x) for x in g.axes.flat[0].get_yticks() * 100]
    g.set_yticklabels(y_labels)

    sns.move_legend(
        g,
        "lower center",
        bbox_to_anchor=(0.5, -0.05),
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
            "Dataset (Train)",
            "Dataset (Eval)",
            "Exact Overlap (# Total Entities)",
            "Partial Overlap (# Total Entities)",
        ]
    ].rename(
        columns={
            "Exact Overlap (# Total Entities)": "Exact Overlap",
            "Partial Overlap (# Total Entities)": "Partial Overlap",
        }
    )

    overlaps_count["Partial Overlap"] = (
        overlaps_count["Partial Overlap"] - overlaps_count["Exact Overlap"]
    )

    overlaps_count = overlaps_count.melt(
        id_vars=["Dataset (Train)", "Dataset (Eval)"],
        var_name="Exposure Type",
        value_name="# Entities seen in Pretraining",
    )

    overlaps_count["Dataset (Train)"] = (
        overlaps_count["Dataset (Train)"].str.split("_").str[0].str.capitalize()
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

    overlaps_count["Dataset (Eval)"] = overlaps_count["Dataset (Eval)"].apply(
        format_dataset_name
    )

    overlaps_count = overlaps_count.rename(
        columns={"Dataset (Train)": "Pretraining Dataset"}
    )

    sns.set_theme(style="whitegrid")
    g = sns.FacetGrid(
        overlaps_count,
        col="Pretraining Dataset",
        height=4,
        aspect=1.2,
    )

    g.map_dataframe(
        sns.barplot,
        "Dataset (Eval)",
        "# Entities seen in Pretraining",
        hue="Exposure Type",
        palette="deep",
    )

    for ax in g.axes.flat:
        ax.set_yscale("symlog")
        ax.set_ylim(0)

    g.add_legend()

    g.set_xticklabels(rotation=30)
    g.set_axis_labels("", "# Entities seen in Pretraining")

    sns.move_legend(
        g,
        "lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=2,
        frameon=False,
    )

    g.tight_layout()
    g.savefig(
        f"overlap_between_datasets_count.png",
        bbox_inches="tight",
    )


def plot_scores_by_label_and_overlap(eval_datasets_scores):
    # Plot Evaluation Dataset Scores
    for metric in eval_datasets_scores["Metric"].unique():
        plot_data = eval_datasets_scores[(eval_datasets_scores["Metric"] == metric)]

        plot_data = plot_data.rename(
            columns={
                "Exposure Type during Training": "Label Seen during Pretraining",
                "Count Evaluation Labels": "# Evaluation Labels",
            }
        )

        plot_data["Dataset (Train)"] = (
            plot_data["Dataset (Train)"].str.split("_").str[0].str.capitalize()
        )

        sns.set_theme(style="whitegrid")
        g = sns.FacetGrid(
            plot_data,
            col="Dataset (Train)",
            col_order=["Ontonotes", "Fewnerd", "Litset", "Pilener"],
            height=4,
            aspect=1.2,
        )

        g.map_dataframe(
            sns.scatterplot,
            "Label Seen (Partial)",
            "Value",
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
        g.add_legend()
        g.fig.text(
            0.5,
            0.0,
            "Number of Evaluation Entities seen during Pretraining",
            ha="center",
        )

        g.tight_layout()
        g.savefig(
            f"scores_by_labels_and_overlap_partial ({metric}).png",
            bbox_inches="tight",
        )

        sns.set_theme(style="whitegrid")
        plot_data["Label Seen during Pretraining"] = plot_data[
            "Label Seen (Exact)"
        ].apply(lambda x: "exact" if x > 0 else "not seen")
        g = sns.FacetGrid(
            plot_data,
            col="Dataset (Train)",
            col_order=["Ontonotes", "Fewnerd", "Litset", "Pilener"],
            height=4,
            aspect=1.2,
        )

        g.map_dataframe(
            sns.scatterplot,
            "Label Seen (Exact)",
            "Value",
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
        g.fig.text(
            0.5,
            0.0,
            "Number of Evaluation Entities seen during Pretraining",
            ha="center",
        )

        g.tight_layout()
        g.savefig(
            f"scores_by_labels_and_overlap_exact ({metric}).png",
            bbox_inches="tight",
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--output_path", type=str, default="/vol/tmp/goldejon/gliner/eval"
    )
    parser.add_argument("--recompute_df", action="store_true")
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

    # Get Overlap Information
    if (
        os.path.exists(os.path.join(args.output_path, "overlap_information.pkl"))
        and not args.recompute_df
    ):
        logger.info("Loading overlap_information.pkl.")
        overlap_information = pd.read_pickle(
            os.path.join(args.output_path, "overlap_information.pkl")
        )
    else:
        logger.info("Get Training Dataset Statistics.")
        train_datasets_stats_aggregated = get_train_datasets_stats(train_datasets_paths)
        logger.info("Get Evaluation Dataset Statistics.")
        eval_datasets_stats_aggregated = get_eval_datasets_stats(eval_datasets_paths)

        logger.info("Compute Overlaps between Train and Evaluation Datasets.")
        overlap_information = train_datasets_stats_aggregated.merge(
            eval_datasets_stats_aggregated,
            how="cross",
            suffixes=[" (Train)", " (Eval)"],
        )
        overlap_information = overlap_information.apply(compute_overlaps, axis=1)

        logger.info("Save overlap_information.pkl.")
        overlap_information.to_pickle(
            os.path.join(args.output_path, "overlap_information.pkl")
        )

    # Get Evaluation Dataset Scores
    if (
        os.path.exists(os.path.join(args.output_path, "eval_datasets_scores.pkl"))
        and not args.recompute_df
    ):
        logger.info("Loading eval_datasets_scores.pkl.")
        eval_datasets_scores = pd.read_pickle(
            os.path.join(args.output_path, "eval_datasets_scores.pkl")
        )
    else:
        logger.info("Get Evaluation Dataset Scores.")
        eval_datasets_scores = get_eval_datasets_scores(
            eval_datasets_paths, train_datasets_paths
        )

        logger.info("Merge Evaluation Dataset Scores with Overlap Information.")
        eval_datasets_scores = eval_datasets_scores.merge(
            overlap_information[
                [
                    "Dataset (Train)",
                    "Dataset (Eval)",
                    "Labels (# Entities) (Eval)",
                    "Exact Overlaps (Labels)",
                    "Partial Overlaps (Labels)",
                    "Exact Overlap (# Entities per Label)",
                    "Partial Overlap (# Entities per Label)",
                ]
            ],
            how="outer",
            on=["Dataset (Train)", "Dataset (Eval)"],
        )

        logger.info("Bin Evaluation Labels")
        eval_datasets_scores = eval_datasets_scores.apply(
            flatten_num_target_labels, axis=1
        )
        bins = [0, 50, 100, 250, 500, float("inf")]
        labels = ["0-50", "50-100", "100-250", "250-500", "500+"]
        eval_datasets_scores["Count Evaluation Labels"] = pd.cut(
            eval_datasets_scores["Count Evaluation Labels"], bins, labels=labels
        )

        logger.info("Identify how labels were exposed during Training.")
        eval_datasets_scores = eval_datasets_scores.apply(
            exposure_type_during_training, axis=1
        )

        logger.info("Extract how many times a label was seen during training.")
        eval_datasets_scores = eval_datasets_scores.apply(count_label_seen, axis=1)

        eval_datasets_scores = eval_datasets_scores.drop(
            columns=[
                "Exact Overlaps (Labels)",
                "Partial Overlaps (Labels)",
                "Labels (# Entities) (Eval)",
                "Exact Overlap (# Entities per Label)",
                "Partial Overlap (# Entities per Label)",
            ]
        )

        logger.info("Save eval_datasets_scores.pkl.")
        eval_datasets_scores.to_pickle(
            os.path.join(args.output_path, "eval_datasets_scores.pkl")
        )

    plot_overlaps_percentage(overlap_information)
    plot_overlaps_count(overlap_information)
    plot_scores_by_label_and_overlap(eval_datasets_scores)
