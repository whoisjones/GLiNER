import os
import json
import random
import logging
from typing import List
from collections import Counter

import pandas as pd
import numpy as np
import seaborn as sns

from data import create_dataset


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


def get_train_labels(train_dataset: str, train_datasets_path: str):
    train_steps = 60000
    batch_size = 4
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
                if train_dataset == "neretrieve_train":
                    labels.append(random.sample(entity[-1], 1)[0])
                else:
                    labels.append(entity[-1])
        labels = labels

    return labels


def get_train_datasets_stats(train_datasets: List, train_datasets_path: str):
    statistics = pd.DataFrame()
    for train_dataset in train_datasets:
        train_labels = get_train_labels(
            train_dataset, train_datasets_path=train_datasets_path
        )
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

        statistics = pd.concat([statistics, df])

    statistics.reset_index(drop=True, inplace=True)

    return statistics


def get_eval_datasets_stats(eval_datasets: List, eval_datasets_path: str):
    statistics = pd.DataFrame()
    for eval_dataset_name in eval_datasets:
        _, _, eval_dataset, _ = create_dataset(
            os.path.join(eval_datasets_path, eval_dataset_name)
        )

        eval_labels = []
        for dp in eval_dataset:
            for entity in dp["ner"]:
                eval_labels.append(entity[-1].lower())

        eval_labels_binary = set(eval_labels)
        eval_labels_count = Counter(eval_labels)

        df = pd.DataFrame(
            {
                "Dataset": [eval_dataset_name],
                "Labels": [eval_labels_binary],
                "Labels (# Entities)": [eval_labels_count],
            }
        )

        statistics = pd.concat([statistics, df])

    statistics.reset_index(drop=True, inplace=True)

    return statistics


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

    display_columns = {
        "ontonotes": "OntoNotes",
        "fewnerd": "FewNERD",
        "neretrieve_train": "NERetrieve",
        "litset": "LitSet",
        "pilener_train": "PileNER",
        "nuner_train": "NuNER",
    }

    overlaps_binary["Dataset (Train)"] = overlaps_binary["Dataset (Train)"].apply(
        lambda x: display_columns.get(x)
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

    column_order = ["OntoNotes", "FewNERD", "NERetrieve", "LitSet", "PileNER", "NuNER"]

    sns.set_theme(style="whitegrid")
    g = sns.FacetGrid(
        overlaps_binary,
        col="Pretraining Dataset",
        height=2.6,
        aspect=1.7,
        col_wrap=3,
        col_order=column_order,
    )

    g.map_dataframe(
        sns.barplot,
        "Dataset (Eval)",
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

    display_columns = {
        "ontonotes": "OntoNotes",
        "fewnerd": "FewNERD",
        "neretrieve_train": "NERetrieve",
        "litset": "LitSet",
        "pilener_train": "PileNER",
        "nuner_train": "NuNER",
    }

    overlaps_count["Dataset (Train)"] = overlaps_count["Dataset (Train)"].apply(
        lambda x: display_columns.get(x)
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

    column_order = ["OntoNotes", "FewNERD", "NERetrieve", "LitSet", "PileNER", "NuNER"]

    sns.set_theme(style="whitegrid")
    g = sns.FacetGrid(
        overlaps_count,
        col="Pretraining Dataset",
        col_order=column_order,
        height=2.6,
        aspect=1.7,
        col_wrap=3,
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
    log_dir = "/vol/tmp/goldejon/gliner/analysis/overlap"
    eval_datasets_path = "/vol/tmp/goldejon/gliner/eval_datasets/NER"
    train_datasets_path = "/vol/tmp/goldejon/gliner/train_datasets"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(os.path.join(log_dir, "overlap_information.pkl")):
        train_datasets = [
            "ontonotes",
            "fewnerd",
            "neretrieve_train",
            "litset",
            "pilener_train",
            "nuner_train",
        ]
        eval_datasets = [
            "mit-movie",
            "mit-restaurant",
            "CrossNER_AI",
            "CrossNER_literature",
            "CrossNER_music",
            "CrossNER_politics",
            "CrossNER_science",
        ]

        logger.info("Get Training Dataset Statistics.")
        train_datasets_stats_aggregated = get_train_datasets_stats(
            train_datasets=train_datasets, train_datasets_path=train_datasets_path
        )
        logger.info("Get Evaluation Dataset Statistics.")
        eval_datasets_stats_aggregated = get_eval_datasets_stats(
            eval_datasets=eval_datasets, eval_datasets_path=eval_datasets_path
        )

        logger.info("Compute Overlaps between Train and Evaluation Datasets.")
        overlap_information = train_datasets_stats_aggregated.merge(
            eval_datasets_stats_aggregated,
            how="cross",
            suffixes=[" (Train)", " (Eval)"],
        )
        overlap_information = overlap_information.apply(compute_overlaps, axis=1)
        overlap_information.to_pickle(os.path.join(log_dir, "overlap_information.pkl"))
    else:
        overlap_information = pd.read_pickle(
            os.path.join(log_dir, "overlap_information.pkl")
        )

    plot_overlaps_percentage(overlap_information)
    plot_overlaps_count(overlap_information)
