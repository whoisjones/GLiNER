import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from data import display_train, get_train_datasets_stats

OUTPUT_PATH = "/vol/tmp/goldejon/gliner/eval_metric_alan"


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
            "angular": [],
            "closest_labels": [],
            "occurrences": [],
        }

        for _, row in tqdm(zeroshot_embeddings.iterrows(), desc="Compute Distance"):
            z_emb = np.stack(row["embedding"]).reshape(1, -1)
            dataset_embeddings = train_embeddings[
                train_embeddings["train_dataset"] == row["train_dataset"]
            ]
            t_emb = np.stack(dataset_embeddings["embedding"])
            similarities = util.cos_sim(z_emb, t_emb)
            angular = 1 - (torch.arccos(similarities) / torch.pi)
            order = similarities.argsort()[0].tolist()[::-1]

            similarities_sorted = [round(x, 4) for x in similarities[0][order].tolist()]
            angular_sorted = [round(x, 4) for x in angular[0][order].tolist()]
            closest_labels = dataset_embeddings.iloc[order]["label"].tolist()
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
            distances["distances"].append(similarities_sorted)
            distances["angular"].append(angular_sorted)
            distances["occurrences"].append(occurrences)
            distances["closest_labels"].append(closest_labels)

        distances_df = pd.DataFrame.from_dict(distances)
        distances_df.to_pickle(f"{OUTPUT_PATH}/distances_cos_sim.pkl")
    else:
        distances_df = pd.read_pickle(f"{OUTPUT_PATH}/distances_cos_sim.pkl")

    return distances_df


def cumsum_until(counts, k):
    cumsum = 0
    result = []

    for count in counts:
        if cumsum + count >= k:
            result.append(k - cumsum)
            break
        else:
            cumsum += count
            result.append(count)

    return result


def compute_metric(distances):
    score_df = {
        "train_dataset": [],
        "eval_benchmark": [],
        "score": [],
        "angular": [],
        "score_type": [],
        "angular_type": [],
        "k": [],
    }
    close_types = [100, 250, 500, 1000, 2500, 5000, 10000]
    for k in close_types:
        for _, row in tqdm(distances.iterrows()):
            counts = cumsum_until(row["occurrences"], k)
            sims = row["distances"][: len(counts)]
            angular = row["angular"][: len(counts)]

            sim_score = np.dot(np.array(sims), np.array(counts)) / k
            angular_score = np.dot(np.array(angular), np.array(counts)) / k

            linear_decay_weights = np.arange(1, k + 1, 1)[::-1] / k
            sim_score_linear_decay = np.dot(
                linear_decay_weights, np.repeat(sims, counts)
            ) / np.sum(linear_decay_weights)
            angular_score_linear_decay = np.dot(
                linear_decay_weights, np.repeat(angular, counts)
            ) / np.sum(linear_decay_weights)

            zipf_weights = 1 / np.arange(1, k + 1, 1)
            sim_score_zipf = np.dot(zipf_weights, np.repeat(sims, counts)) / np.sum(
                zipf_weights
            )
            angular_score_zipf = np.dot(
                zipf_weights, np.repeat(angular, counts)
            ) / np.sum(zipf_weights)

            score_df["train_dataset"].extend(3 * [row["train_dataset"]])
            score_df["eval_benchmark"].extend(3 * [row["eval_benchmark"]])
            score_df["score_type"].append("Weighted Average")
            score_df["score"].append(sim_score)
            score_df["score_type"].append("Weighted Average (Linear Decay)")
            score_df["score"].append(sim_score_linear_decay)
            score_df["score_type"].append("Weighted Average (Zipf)")
            score_df["score"].append(sim_score_zipf)
            score_df["angular_type"].append("Weighted Average")
            score_df["angular"].append(angular_score)
            score_df["angular_type"].append("Weighted Average (Linear Decay)")
            score_df["angular"].append(angular_score_linear_decay)
            score_df["angular_type"].append("Weighted Average (Zipf)")
            score_df["angular"].append(angular_score_zipf)
            score_df["k"].extend(3 * [k])

    score_df = pd.DataFrame.from_dict(score_df)
    score_df.to_pickle(f"{OUTPUT_PATH}/scores.pkl")


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    if not os.path.exists(f"{OUTPUT_PATH}/scores.pkl"):
        train_dataset_stats = get_train_datasets_stats()
        zeroshot_results = get_zeroshot_results()
        train_embeddings, zeroshot_embeddings = compute_embeddings(
            train_dataset_stats, zeroshot_results
        )
        distances = compute_distance(
            train_dataset_stats, train_embeddings, zeroshot_embeddings
        )
        compute_metric(distances)

    score_df = pd.read_pickle(f"{OUTPUT_PATH}/scores.pkl")
    sim_df = (
        score_df[["train_dataset", "k", "score", "score_type"]]
        .groupby(["train_dataset", "k", "score_type"])
        .mean()
        .reset_index()
    )

    sim_df["train_dataset"] = sim_df["train_dataset"].apply(
        lambda x: display_train.get(x)
    )

    sim_df.rename(
        columns={"train_dataset": "Fine-Tuning Dataset", "score": "Familarity"},
        inplace=True,
    )

    angular_df = (
        score_df[["train_dataset", "k", "angular", "angular_type"]]
        .groupby(["train_dataset", "k", "angular_type"])
        .mean()
        .reset_index()
    )

    angular_df["train_dataset"] = angular_df["train_dataset"].apply(
        lambda x: display_train.get(x)
    )

    angular_df.rename(
        columns={"train_dataset": "Fine-Tuning Dataset", "angular": "Familarity"},
        inplace=True,
    )

    sns.set_theme(style="darkgrid", font_scale=1.3)
    g = sns.relplot(
        sim_df,
        x="k",
        y="Familarity",
        col="score_type",
        hue="Fine-Tuning Dataset",
        hue_order=list(display_train.values()),
        kind="line",
        linewidth=3,
        marker="o",
    )
    for item, ax in g.axes_dict.items():
        ax.set_title(item)
    sns.move_legend(g, loc="lower center", bbox_to_anchor=(0.44, -0.13), ncol=6)
    plt.suptitle("Familarity Metric using Cosine Similarity", x=0.44)
    g.tight_layout()
    g.savefig("metric_sim.png")
    plt.clf()

    sns.set_theme(style="darkgrid", font_scale=1.3)
    g = sns.relplot(
        angular_df,
        x="k",
        y="Familarity",
        col="angular_type",
        hue="Fine-Tuning Dataset",
        hue_order=list(display_train.values()),
        kind="line",
        linewidth=3,
        marker="o",
    )
    for item, ax in g.axes_dict.items():
        ax.set_title(item)
    sns.move_legend(g, loc="lower center", bbox_to_anchor=(0.44, -0.13), ncol=6)
    plt.suptitle("Familarity Metric using Angular Similarity", x=0.44)
    g.tight_layout()
    g.savefig("metric_angular.png")
    plt.clf()
