import glob
import pandas as pd


def main():
    paths = glob.glob("/vol/tmp/goldejon/gliner/eval/*/*/results.pkl")

    display_train = {
        "ontonotes": "OntoNotes",
        "fewnerd": "FewNERD",
        "litset": "LitSet",
        "nuner_train": "NuNER",
        "pilener_train": "PileNER",
        "neretrieve_train": "NERRetrieve",
    }

    display_eval = {
        "mit-movie": "Movie",
        "mit-restaurant": "Restaurant",
        "CrossNER_AI": "AI",
        "CrossNER_science": "Science",
        "CrossNER_politics": "Politics",
        "CrossNER_literature": "Literature",
        "CrossNER_music": "Music",
    }

    all_results = pd.DataFrame()
    for path in paths:
        result = pd.read_pickle(path)
        metadata = path.split("/")
        train_dataset = metadata[-3]
        seed = metadata[-2]
        result["train_dataset"] = train_dataset
        result["seed"] = seed
        all_results = pd.concat([all_results, result])

    all_results = all_results[all_results["entity"] == "average"]
    all_results = all_results.reset_index(drop=True)

    all_results["train_dataset"] = all_results["train_dataset"].apply(
        lambda x: display_train[x]
    )

    all_results["eval_benchmark"] = all_results["eval_benchmark"].apply(
        lambda x: display_eval[x]
    )

    mean_std = all_results.groupby(
        ["eval_benchmark", "train_dataset"], as_index=False
    ).agg({"f_score": "mean"})

    average = mean_std.groupby("train_dataset", as_index=False).agg({"f_score": "mean"})
    average["f_score"] = (
        average["f_score"].apply(lambda x: x * 100).round(1).astype(str)
    )
    average["eval_benchmark"] = "Average"

    mean_std["f_score"] = (
        mean_std["f_score"].apply(lambda x: x * 100).round(1).astype(str)
    )

    mean_std = pd.concat([mean_std, average], ignore_index=True)

    mean_std = mean_std.pivot(
        values="f_score", index="train_dataset", columns="eval_benchmark"
    )
    mean_std = mean_std.rename_axis("FT-Dataset", axis=0)

    col_order = [
        "Movie",
        "Restaurant",
        "AI",
        "Science",
        "Politics",
        "Literature",
        "Music",
        "Average",
    ]

    row_order = [
        "OntoNotes",
        "FewNERD",
        "LitSet",
        "NuNER",
        "PileNER",
    ]

    mean_std = mean_std[col_order]
    mean_std = mean_std.reindex(row_order)
    mean_std = mean_std.reset_index().rename_axis(None, axis=1)

    latex = mean_std.style.highlight_max(props="textbf:--rwrap;").hide(axis="index")
    print(latex.to_latex(hrules=True, column_format="l|ccccccc|c"))


if __name__ == "__main__":
    main()
