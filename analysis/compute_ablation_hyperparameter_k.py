from data import get_eval_scores, get_train_datasets_stats
from archive.compute_scores import get_mean_std
from embedding_models import load_model, LabelEmbeddingModel, get_device
import pandas as pd
import numpy as np
from sentence_transformers import util
import torch
from tqdm import tqdm
from data import display_train, display_eval
import seaborn as sns
import matplotlib.pyplot as plt


def batchify(lst, batch_size):
    return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]


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


def compute_embeddings(
    model: LabelEmbeddingModel,
    training_statistics: pd.DataFrame,
    evaluation_results: pd.DataFrame,
    batch_size: int = 32,
) -> pd.DataFrame:
    evaluation_labels = evaluation_results["entity"].unique().tolist()

    tmp_training_statistics = training_statistics[
        ["train_dataset", "train_labels_set_sampled"]
    ]
    training_labels_normalized = tmp_training_statistics.explode(
        "train_labels_set_sampled"
    )
    training_labels = (
        training_labels_normalized["train_labels_set_sampled"].unique().tolist()
    )

    labels = list(set(evaluation_labels + training_labels))
    batches = batchify(labels, batch_size)
    label_embeddings = {}

    with torch.no_grad():
        for batch in tqdm(batches, desc="Embed Labels"):
            batch_embeddings = model.embed(batch)
            label_embeddings.update(batch_embeddings)

    words, embeddings = zip(*label_embeddings.items())
    df = pd.DataFrame(
        {
            "label": words,
            "embedding": embeddings,
        }
    )

    return df


def compute_similarities(
    train_statistics: pd.DataFrame,
    training_embeddings: pd.DataFrame,
    evaluation_embeddings: pd.DataFrame,
):
    sim_df = {
        "train_dataset": [],
        "eval_dataset": [],
        "eval_label": [],
        "similarity": [],
        "angular": [],
        "support": [],
    }

    train_label_counters = {
        dataset: counter
        for dataset, counter in zip(
            *map(
                train_statistics.get, ["train_dataset", "train_labels_counter_sampled"]
            )
        )
    }

    eval_label_dataset_tuples = list(
        zip(*map(evaluation_embeddings.get, ["entity", "eval_dataset"]))
    )

    for train_dataset in training_embeddings["train_dataset"].unique():
        train_embs = training_embeddings[
            training_embeddings["train_dataset"] == train_dataset
        ]

        t_emb = torch.tensor(np.stack(train_embs["embedding"])).to(get_device())
        z_emb = torch.tensor(np.stack(evaluation_embeddings["embedding"])).to(
            get_device()
        )

        similarities = torch.clamp(util.cos_sim(z_emb, t_emb), max=1)
        angular = (1 - (torch.arccos(similarities) / torch.pi)).cpu().numpy()
        similarities = torch.clamp(similarities, min=0).cpu().numpy()

        for idx, (eval_label, eval_dataset) in enumerate(eval_label_dataset_tuples):
            sim_df["train_dataset"].append(train_dataset)
            sim_df["eval_dataset"].append(eval_dataset)
            sim_df["eval_label"].append(eval_label)

            sort_order = np.argsort(similarities[idx])[::-1]

            sorted_train_labels = train_embs["label"].iloc[sort_order].tolist()
            sorted_support = [
                train_label_counters[train_dataset][train_label]
                for train_label in sorted_train_labels
            ]
            sorted_similarities = similarities[idx][sort_order].tolist()
            sorted_angular = angular[idx][sort_order].tolist()

            support_truncated = cumsum_until(sorted_support, 20000)  # space efficiency
            similarity_truncated = sorted_similarities[: len(support_truncated)]
            angular_truncated = sorted_angular[: len(support_truncated)]

            sim_df["similarity"].append(similarity_truncated)
            sim_df["angular"].append(angular_truncated)
            sim_df["support"].append(support_truncated)

    sim_df = pd.DataFrame.from_dict(sim_df)

    return sim_df


def compute_weighted_average(similarities: pd.DataFrame):
    score_df = {
        "train_dataset": [],
        "eval_dataset": [],
        "similarity": [],
        "angular": [],
        "similarity_type": [],
        "angular_type": [],
        "k": [],
    }
    close_types = [100, 250, 500, 1000, 2500, 5000, 10000]
    for k in close_types:
        for _, row in similarities.iterrows():
            counts = cumsum_until(row["support"], k)
            sims = row["similarity"][: len(counts)]
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
            score_df["eval_dataset"].extend(3 * [row["eval_dataset"]])
            score_df["similarity_type"].append("Familarity (Unweighted)")
            score_df["similarity"].append(sim_score)
            score_df["similarity_type"].append("Familarity (Linear Decay)")
            score_df["similarity"].append(sim_score_linear_decay)
            score_df["similarity_type"].append("Familarity")
            score_df["similarity"].append(sim_score_zipf)
            score_df["angular_type"].append("Familarity (Unweighted)")
            score_df["angular"].append(angular_score)
            score_df["angular_type"].append("Familarity (Linear Decay)")
            score_df["angular"].append(angular_score_linear_decay)
            score_df["angular_type"].append("Familarity")
            score_df["angular"].append(angular_score_zipf)
            score_df["k"].extend(3 * [k])

    score_df = pd.DataFrame.from_dict(score_df)
    return score_df


if __name__ == "__main__":
    training_statistics = get_train_datasets_stats(base_path="/vol/tmp/goldejon/gliner")
    evaluation_statistics = get_eval_scores(base_path="/vol/tmp/goldejon/gliner")
    evaluation_statistics = evaluation_statistics[
        ["entity", "eval_dataset"]
    ].drop_duplicates()
    evaluation_results = get_mean_std(base_path="/vol/tmp/goldejon/gliner")
    evaluation_results = evaluation_results[["FT-Dataset", "Average"]]
    model = load_model("sentence-transformers/all-mpnet-base-v2")
    label_embeddings = compute_embeddings(
        model, training_statistics, evaluation_statistics
    )

    embedded_evaluation = pd.merge(
        evaluation_statistics,
        label_embeddings,
        left_on="entity",
        right_on="label",
        how="inner",
    ).drop(columns="label")

    train_df = training_statistics[
        ["train_dataset", "train_labels_set_sampled"]
    ].explode("train_labels_set_sampled")

    embedded_training = pd.merge(
        train_df,
        label_embeddings,
        left_on="train_labels_set_sampled",
        right_on="label",
        how="inner",
    ).drop(columns="train_labels_set_sampled")

    similarities = compute_similarities(
        training_statistics, embedded_training, embedded_evaluation
    )

    metrics = compute_weighted_average(similarities)

    metrics["train_dataset"] = metrics["train_dataset"].replace(display_train)
    metrics["eval_dataset"] = metrics["eval_dataset"].replace(display_eval)

    metrics = (
        metrics[["train_dataset", "k", "similarity", "similarity_type"]]
        .groupby(["train_dataset", "k", "similarity_type"])
        .mean()
        .reset_index()
    )

    metrics = metrics[
        (metrics["train_dataset"] != "OntoNotes")
        & (metrics["train_dataset"] != "FewNERD")
    ]

    metrics.rename(
        columns={
            "train_dataset": "Fine-Tuning Dataset",
            "similarity": "Familarity",
        },
        inplace=True,
    )

    sns.set_theme(style="whitegrid", palette="plasma", font_scale=1.3)
    g = sns.relplot(
        metrics,
        x="k",
        y="Familarity",
        col="similarity_type",
        col_order=[
            "Familarity",
            "Familarity (Linear Decay)",
            "Familarity (Unweighted)",
        ],
        hue="Fine-Tuning Dataset",
        hue_order=[
            x for x in list(display_train.values()) if x not in ["OntoNotes", "FewNERD"]
        ],
        kind="line",
        linewidth=4,
        marker="d",
        markersize=10,
    )
    for item, ax in g.axes_dict.items():
        ax.set_title(item)
    sns.move_legend(g, loc="lower center", bbox_to_anchor=(0.44, -0.13), ncol=6)
    g.tight_layout()
    g.savefig("ablation_hyperparameter_k.png", dpi=640)
    plt.close("all")

    print()
