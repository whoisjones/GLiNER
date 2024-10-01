import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from compute_scores import get_mean_std

from data import (
    compute_overlaps,
    display_train,
    get_eval_datasets_stats,
    get_train_datasets_stats,
)

if __name__ == "__main__":
    mean_std = get_mean_std("/vol/tmp/goldejon/gliner/paper_data")
    mean_std = mean_std[["FT-Dataset", "Average"]]

    train_statistics = get_train_datasets_stats(base_path="/vol/tmp/goldejon/gliner")
    train_statistics = train_statistics[train_statistics["train_dataset"] != "fewnerd"]
    train_statistics = train_statistics[
        train_statistics["train_dataset"] != "ontonotes"
    ]
    eval_statistics = get_eval_datasets_stats(base_path="/vol/tmp/goldejon/gliner")

    train_statistics["total_entities"] = train_statistics[
        "train_labels_counter_sampled"
    ].apply(lambda c: round(sum(c.values()), 2))

    overlap_information = train_statistics.merge(
        eval_statistics,
        how="cross",
    )

    overlap_information = overlap_information.apply(compute_overlaps, axis=1)

    overlap_information = (
        overlap_information[
            ["train_dataset", "exact_overlap_percentage", "total_entities"]
        ]
        .groupby("train_dataset")
        .agg({"exact_overlap_percentage": "mean", "total_entities": "mean"})
    )

    overlap_information.reset_index(inplace=True)

    overlap_information["train_dataset"] = overlap_information["train_dataset"].apply(
        lambda x: display_train.get(x)
    )

    overlap_information["exact_overlap_percentage"] = (
        overlap_information["exact_overlap_percentage"]
        .apply(lambda x: x * 100)
        .round(1)
        .astype(str)
    )

    score_label = "Avg. Zero-Shot Results"
    dataset_label = "Fine-Tuning Datasets"
    overlap_label = "% Overlapping Entity Types \n w/ Evaluation Datasets"
    entity_label = "# Entity Mentions \n in Fine-Tuning Dataset"

    overlap_information.rename(
        columns={
            "train_dataset": dataset_label,
            "exact_overlap_percentage": overlap_label,
            "total_entities": entity_label,
        },
        inplace=True,
    )

    mean_std.rename(columns={"FT-Dataset": dataset_label}, inplace=True)

    table = mean_std.merge(overlap_information, how="inner", on=dataset_label)
    table.rename(columns={"Average": score_label}, inplace=True)
    table[score_label] = pd.to_numeric(table[score_label])
    table[overlap_label] = pd.to_numeric(table[overlap_label])
    table[entity_label] = pd.to_numeric(table[entity_label])

    plt.rcParams.update({"font.size": 14})
    fig, ax1 = plt.subplots()
    x = np.arange(len(table))
    width = 0.4

    ax1.bar(
        x - 0.2, table[overlap_label], width, label=overlap_label, color="navajowhite"
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(table[dataset_label], rotation=45, ha="right")
    ax1.set_xlabel(dataset_label)
    ax1.set_ylim([0, 100])
    ax1.set_yticks(np.arange(0, 101, 20))
    ax1_ylabel = ax1.set_ylabel(overlap_label, labelpad=15)
    ax1_ylabel.set_bbox(dict(facecolor="navajowhite", edgecolor="white"))

    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, table[entity_label], width, label=entity_label, color="thistle")
    ax2.set_xticks(x)
    ax2.set_xticklabels(table[dataset_label], rotation=45, ha="right")
    ax2.set_yscale("log")
    ax2.set_ylim([10**5, 10**7])
    ax2.set_yticks([10**5, 10**6, 10**7])
    ax2_ylabel = ax2.set_ylabel(entity_label, labelpad=15)
    ax2_ylabel.set_bbox(dict(facecolor="thistle", edgecolor="none"))

    ax3 = ax1.twinx()
    ax3.plot(
        x,
        table[score_label],
        color="tab:red",
        label=score_label,
        linewidth=3,
        marker="o",
        markersize=10,
    )
    for i, (xi, yi) in enumerate(zip(x, table[score_label])):
        ax3.text(
            xi - 0.25,
            yi - 2,
            str(yi),
            ha="left",
            size="medium",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                edgecolor="tab:red",
                linewidth=3,
                alpha=0.96,
                pad=0.2,
            ),
        )
    ax3.set_xticks(x)
    ax3.set_xticklabels(table[dataset_label])
    ax3.get_yaxis().set_visible(False)
    ax3.set_ylim([0, 100])

    lines, labels = ax3.get_legend_handles_labels()
    ax3.legend(
        lines, labels, loc="center", bbox_to_anchor=(0.5, 1.075), ncol=1, frameon=False
    )

    plt.tight_layout()
    plt.savefig("test.png", dpi=620)
    plt.close()
