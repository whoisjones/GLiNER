import glob

import numpy as np
import pandas as pd
from compute_scores import get_mean_std

from data import display_train


def print_latex():
    mean_std = get_mean_std("/vol/tmp/goldejon/gliner")
    mean_std = mean_std[mean_std["FT-Dataset"] != "OntoNotes"]
    mean_std = mean_std[mean_std["FT-Dataset"] != "FewNERD"]
    mean_std = mean_std[["FT-Dataset", "Average"]].T
    col_order = mean_std.iloc[0]
    mean_std.columns = col_order
    mean_std.drop("FT-Dataset", inplace=True)

    runs = glob.glob("logs/*")
    model_names = {
        "Average": "\diameter~Zero-Shot F1",
        "crawl-300d-2M.vec": "fasttext-crawl-300d-2M",
        "wiki-news-300d-1M.vec": "fasttext-wiki-news-300d-1M",
        "glove.6B.300d.txt": "glove-6B-300d",
        "bert-base-uncased": "bert-base-uncased",
        "distilbert-base-uncased": "distilbert-base-uncased",
        "sentence-transformers/all-mpnet-base-v2": "all-mpnet-base-v2",
        "sentence-transformers/all-MiniLM-L6-v2": "all-miniLM-L6-v2",
    }
    for run in runs:
        files = glob.glob(run + "/*")
        for file in files:
            if file.endswith(".pkl"):
                df = pd.read_pickle(file)
            else:
                with open(file) as f:
                    lines = f.readlines()
                parts = lines[0].split(":")
                model = parts[1].strip()
        if "deberta" in model or "840B" in model:
            continue
        df = df[df["k"] == 1000]
        df = df[df["similarity_type"] == "Weighted Average"]
        df["train_dataset"] = df["train_dataset"].apply(lambda x: display_train[x])
        output = (
            df[["train_dataset", "similarity"]]
            .groupby(["train_dataset"])
            .mean()
            .reset_index()
            .rename(columns={"similarity": model})
        )
        output[model] = output[model].map("{:,.3f}".format)
        output = output.T
        output.columns = output.iloc[0]
        output = output.drop("train_dataset")
        output = output.reindex(col_order, axis=1)
        mean_std = pd.concat([mean_std, output])

    mean_std = mean_std.rename(index=model_names)
    mean_std = mean_std.reindex(model_names.values())
    mean_std.columns = pd.MultiIndex.from_tuples(
        [("Fine-Tuning on:", col) for col in mean_std.columns]
    )

    for idx, row in mean_std.iterrows():
        if idx == "\diameter~Zero-Shot F1":
            gold_order = np.argsort(row.values)
            continue
        pred_order = np.argsort(row.values)
        mean_std.loc[idx] = [
            "".join(x)
            for x in zip(
                [
                    "\\" if p == gold_order[i] else ""
                    for i, (x, p) in enumerate(zip(row.values, pred_order))
                ],
                [
                    f"{'underline{' if p == gold_order[i] else ''}{x}{'}' if p == gold_order[i] else ''}"
                    for i, (x, p) in enumerate(zip(row.values, pred_order))
                ],
            )
        ]

    print(
        mean_std.to_latex(
            escape=False,
            multicolumn_format="c",
            column_format="l" + "c" * len(mean_std.columns),
        )
    )


if __name__ == "__main__":
    print_latex()
