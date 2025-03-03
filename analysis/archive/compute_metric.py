import glob
import math
import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from data import display_train, get_train_datasets_stats

OUTPUT_PATH = "/vol/tmp/goldejon/gliner/eval_metric"


def get_zeroshot_results(average=False):
    paths = glob.glob("/vol/tmp/goldejon/gliner/eval/*/*/results.pkl")

    all_results = pd.DataFrame()
    for path in paths:
        result = pd.read_pickle(path)
        metadata = path.split("/")
        train_dataset = metadata[-3]
        seed = metadata[-2]
        result["train_dataset"] = train_dataset
        result["seed"] = seed
        all_results = pd.concat([all_results, result])

    if average:
        all_results = all_results[all_results["entity"] == "average"]
    else:
        all_results = all_results[all_results["entity"] != "average"]
    all_results = all_results.reset_index(drop=True)
    return all_results


def compute_embeddings(train_dataset_stats, zeroshot_results):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    model.to("cuda")

    zeroshot_labels = zeroshot_results["entity"].unique()
    train_labels_df = train_dataset_stats[["train_dataset", "train_labels_set_sampled"]]
    train_labels_normalized = train_labels_df.explode("train_labels_set_sampled")
    train_labels = train_labels_normalized["train_labels_set_sampled"].unique().tolist()

    batch_size = 16
    zeroshot_embeddings = []
    train_label_embeddings = []

    if not os.path.exists(f"{OUTPUT_PATH}/zeroshot_embeddings.pkl"):
        for i in tqdm(
            range(0, len(zeroshot_labels), batch_size), desc="Zeroshot Embedding"
        ):
            emb = (
                model.encode(
                    zeroshot_labels[i : i + batch_size], convert_to_tensor=True
                )
                .detach()
                .cpu()
                .numpy()
            )
            zeroshot_embeddings.append(emb)

        embeddings = np.concatenate(zeroshot_embeddings)
        zeroshot_embedding_df = {"label": [], "embedding": []}
        for i in range(len(zeroshot_labels)):
            zeroshot_embedding_df["label"].append(zeroshot_labels[i])
            zeroshot_embedding_df["embedding"].append(embeddings[i])
        zeroshot_embedding_df = pd.DataFrame.from_dict(zeroshot_embedding_df)

        output_zeroshot = pd.merge(
            zeroshot_results,
            zeroshot_embedding_df,
            left_on="entity",
            right_on="label",
            how="inner",
        ).drop(columns="label")
        output_zeroshot.to_pickle(f"{OUTPUT_PATH}/zeroshot_embeddings.pkl")
    else:
        output_zeroshot = pd.read_pickle(f"{OUTPUT_PATH}/zeroshot_embeddings.pkl")

    if not os.path.exists(f"{OUTPUT_PATH}/train_label_embeddings.pkl"):
        for i in tqdm(
            range(0, len(train_labels), batch_size), desc="Train Label Embedding"
        ):
            emb = (
                model.encode(train_labels[i : i + batch_size], convert_to_tensor=True)
                .detach()
                .cpu()
                .numpy()
            )
            train_label_embeddings.append(emb)

        embeddings = np.concatenate(train_label_embeddings)
        train_label_embedding_df = {"label": [], "embedding": []}
        for i in range(len(train_labels)):
            train_label_embedding_df["label"].append(train_labels[i])
            train_label_embedding_df["embedding"].append(embeddings[i])
        train_label_embedding_df = pd.DataFrame.from_dict(train_label_embedding_df)

        output_train = pd.merge(
            train_labels_normalized,
            train_label_embedding_df,
            left_on="train_labels_set_sampled",
            right_on="label",
            how="inner",
        ).drop(columns="train_labels_set_sampled")
        output_train.to_pickle(f"{OUTPUT_PATH}/train_label_embeddings.pkl")
    else:
        output_train = pd.read_pickle(f"{OUTPUT_PATH}/train_label_embeddings.pkl")

    return output_train, output_zeroshot


def compute_distance(train_stats, train_embeddings, zeroshot_embeddings):
    if not os.path.exists(f"{OUTPUT_PATH}/distances_cos_sim.pkl"):
        distances = {
            "train_dataset": [],
            "eval_benchmark": [],
            "label": [],
            "score": [],
            "distances": [],
            "closest_labels": [],
            "occurrences": [],
        }

        for _, row in tqdm(zeroshot_embeddings.iterrows(), desc="Compute Distance"):
            z_emb = np.stack(row["embedding"]).reshape(1, -1)
            dataset_embeddings = train_embeddings[
                train_embeddings["train_dataset"] == row["train_dataset"]
            ]
            t_emb = np.stack(dataset_embeddings["embedding"])
            sim = util.cos_sim(z_emb, t_emb)
            top_k = sim.argsort()[0].tolist()[::-1]

            closest_similarities = sim[0][top_k]
            closest_labels = dataset_embeddings.iloc[top_k]["label"].tolist()
            train_stats[train_stats["train_dataset"] == row["train_dataset"]][
                "train_labels_counter_sampled"
            ]
            counter = train_stats[train_stats["train_dataset"] == row["train_dataset"]][
                "train_labels_counter_sampled"
            ].iloc[0]
            occurrences = [counter[label] for label in closest_labels]

            distances["train_dataset"].append(row["train_dataset"])
            distances["eval_benchmark"].append(row["eval_benchmark"])
            distances["label"].append(row["entity"])
            distances["score"].append(row["f_score"])
            distances["distances"].append(closest_similarities)
            distances["occurrences"].append(occurrences)
            distances["closest_labels"].append(closest_labels)

        distances_df = pd.DataFrame.from_dict(distances)
        distances_df.to_pickle(f"{OUTPUT_PATH}/distances_cos_sim.pkl")
    else:
        distances_df = pd.read_pickle(f"{OUTPUT_PATH}/distances_cos_sim.pkl")

    return distances_df


def compute_metric_top_p(row, p):
    mask = row["distances"] >= p
    distances = row["distances"][mask]
    occurrences = np.array(row["occurrences"])[mask]
    if distances.nelement() == 0:
        return 0
    else:
        metric = sum(distances * occurrences) / sum(occurrences)
        return metric


def compute_metric_top_k(row, k):
    distances = row["distances"][:k]
    occurrences = np.array(row["occurrences"])[:k]
    metric = sum(distances * occurrences) / sum(occurrences)
    return metric


def compute_support_top_p(row, p):
    mask = row["distances"] >= p
    distances = row["distances"][mask]
    occurrences = np.array(row["occurrences"])[mask]
    if distances.nelement() == 0:
        return 0
    else:
        support = math.log10(sum(occurrences))
        return support


def compute_support_top_k(row, k):
    distances = row["distances"][:k]
    occurrences = np.array(row["occurrences"])[:k]
    if distances.nelement() == 0:
        return 0
    else:
        support = math.log10(sum(occurrences))
        return support


def init_result():
    zeroshot_results_avg = get_zeroshot_results(average=True)
    result = (
        zeroshot_results_avg[["train_dataset", "f_score"]]
        .groupby("train_dataset")
        .mean()
        .apply(lambda x: x * 100)
        .round(2)
        .astype(str)
    )
    return result


def compute_top_k(distances):
    result = init_result()

    for k in [1, 3, 5, 10]:
        dataset_distances = distances[["train_dataset", "distances", "occurrences"]]
        dataset_distances["sim"] = dataset_distances.apply(
            lambda x: compute_metric_top_k(x, k), axis=1
        )
        dataset_distances["sup"] = dataset_distances.apply(
            lambda x: compute_support_top_k(x, k), axis=1
        )
        dataset_distances.drop(columns=["distances", "occurrences"], inplace=True)

        dataset_distances = dataset_distances.groupby("train_dataset").mean()
        dataset_distances["sim"] = dataset_distances["sim"].apply(
            lambda x: "{:.2f}".format(x)
        )
        dataset_distances["sup"] = dataset_distances["sup"].apply(
            lambda x: "{:.2f}".format(x)
        )
        dataset_distances[f"Top-{k} S@S"] = (
            dataset_distances["sim"].astype(str)
            + "@"
            + dataset_distances["sup"].astype(str)
        )
        dataset_distances.drop(columns=["sim", "sup"], inplace=True)

        result = result.join(dataset_distances)

    result = result.reset_index().rename(
        columns={"f_score": "Avg. F1 Score", "train_dataset": "FT-Dataset"}
    )
    result["FT-Dataset"] = result["FT-Dataset"].apply(lambda x: display_train[x])
    mapping = {v: k for k, v in dict(enumerate(display_train.values())).items()}
    result = result.sort_values(by="FT-Dataset", key=lambda x: x.map(mapping))
    latex = result.style.hide(axis="index")
    print(latex.to_latex(hrules=True, column_format="l|ccccc"))


def compute_top_p(distances):
    result = init_result()

    for p in [0.9, 0.75, 0.5]:
        dataset_distances = distances[
            ["train_dataset", "distances", "occurrences"]
        ].copy()
        dataset_distances["sim"] = dataset_distances.apply(
            lambda x: compute_metric_top_p(x, p), axis=1
        )
        dataset_distances["sup"] = dataset_distances.apply(
            lambda x: compute_support_top_p(x, p), axis=1
        )
        dataset_distances.drop(columns=["distances", "occurrences"], inplace=True)

        dataset_distances = dataset_distances.groupby("train_dataset").mean()
        dataset_distances["sim"] = dataset_distances["sim"].apply(
            lambda x: "{:.2f}".format(x)
        )
        dataset_distances["sup"] = dataset_distances["sup"].apply(
            lambda x: "{:.2f}".format(x)
        )
        dataset_distances[f"Top-{p} S@S"] = (
            dataset_distances["sim"].astype(str)
            + "@"
            + dataset_distances["sup"].astype(str)
        )
        dataset_distances.drop(columns=["sim", "sup"], inplace=True)

        result = result.join(dataset_distances)

    result = result.reset_index().rename(
        columns={"f_score": "Avg. F1 Score", "train_dataset": "FT-Dataset"}
    )
    result["FT-Dataset"] = result["FT-Dataset"].apply(lambda x: display_train[x])
    mapping = {v: k for k, v in dict(enumerate(display_train.values())).items()}
    result = result.sort_values(by="FT-Dataset", key=lambda x: x.map(mapping))
    latex = result.style.hide(axis="index")
    print(latex.to_latex(hrules=True, column_format="l|cccccc"))


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    train_dataset_stats = get_train_datasets_stats()
    zeroshot_results = get_zeroshot_results()
    train_embeddings, zeroshot_embeddings = compute_embeddings(
        train_dataset_stats, zeroshot_results
    )
    distances = compute_distance(
        train_dataset_stats, train_embeddings, zeroshot_embeddings
    )
    compute_top_k(distances)
    compute_top_p(distances)
