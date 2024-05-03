from compute_scores import get_mean_std
from compute_overlaps import (
    get_train_datasets_stats,
    get_eval_datasets_stats,
    compute_overlaps,
    display_train,
)
import pandas as pd

if __name__ == "__main__":
    mean_std = get_mean_std()
    mean_std = mean_std[["FT-Dataset", "Average"]]

    train_statistics = get_train_datasets_stats()
    eval_statistics = get_eval_datasets_stats()

    overlap_information = train_statistics.merge(
        eval_statistics,
        how="cross",
    )

    overlap_information = overlap_information.apply(compute_overlaps, axis=1)

    overlap_information = (
        overlap_information[
            ["train_dataset", "exact_overlap_percentage", "partial_overlap_percentage"]
        ]
        .groupby("train_dataset")
        .agg({"exact_overlap_percentage": "mean", "partial_overlap_percentage": "mean"})
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

    overlap_information["partial_overlap_percentage"] = (
        overlap_information["partial_overlap_percentage"]
        .apply(lambda x: x * 100)
        .round(1)
        .astype(str)
    )

    overlap_information.rename(
        columns={
            "train_dataset": "FT-Dataset",
            "exact_overlap_percentage": "Exact",
            "partial_overlap_percentage": "Partial",
        },
        inplace=True,
    )

    table = mean_std.merge(overlap_information, how="inner", on="FT-Dataset")
    table.rename(columns={"Average": "Avg. Zero-Shot"}, inplace=True)
    index_top_level = [
        "FT-Dataset",
        "Avg. Zero-Shot",
        "\% Labels in FT",
        "\% Labels in FT",
    ]
    index_bottom_level = ["", "Transfer", "Exact", "Partial"]
    table.columns = pd.MultiIndex.from_tuples(
        list(zip(index_top_level, index_bottom_level))
    )

    latex = table.style.hide(axis="index")
    print(latex.to_latex(hrules=True, column_format="lc|cc"))
