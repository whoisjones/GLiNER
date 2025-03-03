import sys

sys.path.append("/vol/fob-vol7/mi18/goldejon/GLiNER")

import glob
import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sentence_transformers import SentenceTransformer
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from tqdm import tqdm

from data import display_eval, display_train, get_train_datasets_stats
from gliner import GLiNER

OUTPUT_PATH = "/vol/tmp/goldejon/gliner/eval_metric_max"


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


def compute_gliner(train_datasets, eval_benchmarks):
    unique_zeroshots = eval_benchmarks[["entity", "eval_benchmark"]].drop_duplicates()

    batch_size = 32

    configs = {
        "ontonotes": ["123/model_60000", "234/model_60000", "345/model_50000"],
        "fewnerd": ["123/model_60000", "234/model_60000", "345/model_60000"],
        "neretrieve_train": [
            # "123/model_60000",
            "234/model_60000",
            "345/model_60000",
        ],
        "litset": ["123/model_60000", "234/model_60000", "345/model_60000"],
        "nuner_train": ["123/model_60000", "234/model_60000", "345/model_60000"],
        "pilener_train": ["123/model_60000", "234/model_60000", "345/model_60000"],
    }
    with torch.no_grad():
        for eval_benchmark in unique_zeroshots["eval_benchmark"].unique():
            target_labels = unique_zeroshots[
                unique_zeroshots["eval_benchmark"] == eval_benchmark
            ]["entity"].tolist()

            for train_dataset, model_paths in configs.items():
                train_label_counter = train_datasets[
                    train_datasets["train_dataset"] == train_dataset
                ]["train_labels_counter_sampled"].values[0]

                train_labels = list(train_label_counter.keys())
                train_labels_count = list(train_label_counter.values())

                for model_path in model_paths:
                    seed = model_path.split("/")[0]
                    model_ckpt = f"/vol/tmp/goldejon/gliner/logs/deberta-v3-large/{train_dataset}/{model_path}"

                    if os.path.exists(
                        f"{OUTPUT_PATH}/scores_{train_dataset}_{seed}_{eval_benchmark}_gliner.pkl"
                    ):
                        continue

                    output_file = {
                        "train_label": [],
                        "support": [],
                        "target_label_x": [],
                        "target_label_y": [],
                        "score": [],
                    }

                    model = GLiNER.from_pretrained(model_ckpt)
                    model.eval()
                    if torch.cuda.is_available():
                        model = model.to("cuda")

                    target_prompt, len_target_prompt = gliner_prompt(
                        target_labels,
                        sep_token=model.sep_token,
                        ent_token=model.entity_token,
                    )
                    target_embeddings = model.token_rep_layer(
                        target_prompt, torch.tensor(len_target_prompt)
                    )
                    target_embeddings = target_embeddings["embeddings"][
                        :, :-1:2
                    ].squeeze()
                    target_embeddings = model.prompt_rep_layer(target_embeddings)

                    for idx in tqdm(range(0, len(train_labels), batch_size)):
                        batch = train_labels[idx : idx + batch_size]
                        batch_count = train_labels_count[idx : idx + batch_size]
                        train_prompt, len_train_prompt = gliner_prompt(
                            batch,
                            sep_token=model.sep_token,
                            ent_token=model.entity_token,
                        )
                        train_embeddings = model.token_rep_layer(
                            train_prompt, torch.tensor(len_train_prompt)
                        )
                        train_embeddings = train_embeddings["embeddings"][
                            :, :-1:2
                        ].squeeze()
                        train_embeddings = model.prompt_rep_layer(train_embeddings)

                        output = compute_scores(train_embeddings, target_embeddings)
                        output_triu, label_x, label_y = output_to_triu(
                            output, target_labels
                        )

                        output_file["train_label"].extend(
                            list(np.repeat(batch, len(label_x)))
                        )
                        output_file["support"].extend(
                            [int(x) for x in np.repeat(batch_count, len(label_x))]
                        )
                        output_file["target_label_x"].extend(
                            list(np.tile(label_x, len(batch)))
                        )
                        output_file["target_label_y"].extend(
                            list(np.tile(label_y, len(batch)))
                        )
                        output_file["score"].extend(
                            [round(float(x), 4) for x in output_triu.flatten().tolist()]
                        )

                    result_df = pd.DataFrame.from_dict(output_file)
                    result_df.to_pickle(
                        f"{OUTPUT_PATH}/scores_{train_dataset}_{seed}_{eval_benchmark}_gliner.pkl"
                    )


def gliner_prompt(label_prompts, sep_token, ent_token):
    gliner_prompt = []
    for label in label_prompts:
        gliner_prompt.append([ent_token, label, sep_token])
    return gliner_prompt, [len(prompt) for prompt in gliner_prompt]


def compute_sentence_transformers(train_datasets, eval_benchmarks):
    unique_zeroshots = eval_benchmarks[["entity", "eval_benchmark"]].drop_duplicates()

    model = SentenceTransformer("all-MiniLM-L6-v2")
    model.to("cuda")

    batch_size = 32

    with torch.no_grad():
        for eval_benchmark in unique_zeroshots["eval_benchmark"].unique():
            target_labels = unique_zeroshots[
                unique_zeroshots["eval_benchmark"] == eval_benchmark
            ]["entity"].tolist()

            target_embeddings = model.encode(target_labels, convert_to_tensor=True)

            for train_dataset in train_datasets["train_dataset"].unique():
                if os.path.exists(
                    f"{OUTPUT_PATH}/scores_{train_dataset}_{eval_benchmark}_sentence_transformers.pkl"
                ):
                    continue

                output_file = {
                    "train_label": [],
                    "support": [],
                    "target_label_x": [],
                    "target_label_y": [],
                    "score": [],
                }

                train_label_counter = train_datasets[
                    train_datasets["train_dataset"] == train_dataset
                ]["train_labels_counter_sampled"].values[0]

                train_labels = list(train_label_counter.keys())
                train_labels_count = list(train_label_counter.values())

                for idx in tqdm(range(0, len(train_labels), batch_size)):
                    batch = train_labels[idx : idx + batch_size]
                    batch_count = train_labels_count[idx : idx + batch_size]
                    train_embeddings = model.encode(batch, convert_to_tensor=True)

                    output = compute_scores(train_embeddings, target_embeddings)
                    output_triu, label_x, label_y = output_to_triu(
                        output, target_labels
                    )

                    output_file["train_label"].extend(
                        list(np.repeat(batch, len(label_x)))
                    )
                    output_file["support"].extend(
                        [int(x) for x in np.repeat(batch_count, len(label_x))]
                    )
                    output_file["target_label_x"].extend(
                        list(np.tile(label_x, len(batch)))
                    )
                    output_file["target_label_y"].extend(
                        list(np.tile(label_y, len(batch)))
                    )
                    output_file["score"].extend(
                        [round(float(x), 4) for x in output_triu.flatten().tolist()]
                    )

                result_df = pd.DataFrame.from_dict(output_file)
                result_df.to_pickle(
                    f"{OUTPUT_PATH}/scores_{train_dataset}_{eval_benchmark}_sentence_transformers.pkl"
                )


def compute_scores(train_embeddings, target_embeddings):
    similarities = torch.acos(
        torch.clamp(
            torch.nn.functional.cosine_similarity(
                train_embeddings.unsqueeze(1),
                target_embeddings.unsqueeze(0),
                dim=-1,
            ),
            min=-1.0,
            max=1.0,
        )
    )
    pairwise_distances = similarities.unsqueeze(1) - similarities.unsqueeze(-1)
    norm = torch.acos(pairwise_cosine_similarity(target_embeddings, target_embeddings))
    output = torch.abs(pairwise_distances / norm)

    return output


def output_to_triu(output, target_labels):
    output = torch.triu(output, diagonal=1)
    indices = torch.triu_indices(len(target_labels), len(target_labels)).to("cuda")
    output = output[:, indices[0], indices[1]].cpu().numpy()
    flat_indices = (indices[0] * len(target_labels) + indices[1]).cpu().numpy()
    target_pairs = np.array(list(product(target_labels, target_labels)))[
        flat_indices
    ].tolist()
    target_label_x, target_label_y = zip(*target_pairs)
    return output, target_label_x, target_label_y


def plot_confusion_matrices(train_datasets, zeroshot_datasets):
    overall_results = pd.DataFrame()
    for model in ["sentence_transformers", "gliner"]:
        for zeroshot_dataset in zeroshot_datasets:
            outputs = {}
            for train_dataset in train_datasets:
                if model == "sentence_transformers":
                    seeds = [""]
                else:
                    seeds = ["_123", "_234", "_345"]
                for seed in seeds:
                    path_template = (
                        f"scores_{train_dataset}{seed}_{zeroshot_dataset}_{model}.pkl"
                    )

                    if not os.path.exists(f"{OUTPUT_PATH}/{path_template}"):
                        continue

                    scores = pd.read_pickle(f"{OUTPUT_PATH}/{path_template}")
                    scores["score"] = scores["score"] * scores["support"]

                    scores_normed = scores_normalized_by_support(scores)
                    normed_conf_matrix = create_conf_matrix(scores_normed)

                    if not outputs.get(train_dataset):
                        outputs[train_dataset] = {}
                    outputs[train_dataset]["normed"] = normed_conf_matrix

                    scores_log_sum = scores_log(scores)
                    log_sum_conf_matrix = create_conf_matrix(scores_log_sum)
                    outputs[train_dataset]["log_sum"] = log_sum_conf_matrix

                    scores_normed["train_dataset"] = train_dataset
                    scores_normed["seed"] = seed
                    scores_normed["zeroshot_dataset"] = zeroshot_dataset
                    scores_normed["model"] = model
                    scores_normed["agg_type"] = "normed"

                    scores_log_sum["train_dataset"] = train_dataset
                    scores_log_sum["seed"] = seed
                    scores_log_sum["zeroshot_dataset"] = zeroshot_dataset
                    scores_log_sum["model"] = model
                    scores_log_sum["agg_type"] = "log_sum"

                    overall_results = pd.concat(
                        [overall_results, scores_normed, scores_log_sum]
                    )

            matrices_normed, matrices_log_sum = [], []
            for train_dataset_uncased, train_dataset_cased in display_train.items():
                matrices_normed.append(
                    (
                        outputs.get(train_dataset_uncased).get("normed"),
                        train_dataset_cased,
                    )
                )
                matrices_log_sum.append(
                    (
                        outputs.get(train_dataset_uncased).get("log_sum"),
                        train_dataset_cased,
                    )
                )

            plot(
                matrices_normed,
                agg_type="normed",
                model=model,
                zeroshot_dataset=zeroshot_dataset,
                seed=seed,
            )
            plot(
                matrices_log_sum,
                agg_type="log_sum",
                model=model,
                zeroshot_dataset=zeroshot_dataset,
                seed=seed,
            )

    overall_results.to_pickle(f"{OUTPUT_PATH}/overall_results.pkl")


def scores_normalized_by_support(scores_df):
    max_value = (
        scores_df[["train_label", "support"]]
        .groupby("train_label")
        .mean()["support"]
        .sum()
    )
    scores_agg = (
        scores_df[["target_label_x", "target_label_y", "score"]]
        .groupby(["target_label_x", "target_label_y"])
        .sum()
        .reset_index()
    )
    scores_agg["score"] = scores_agg["score"] / max_value
    return scores_agg


def scores_log(scores_df):
    scores_agg = (
        scores_df[["target_label_x", "target_label_y", "score"]]
        .groupby(["target_label_x", "target_label_y"])
        .sum()
        .reset_index()
    )
    scores_agg["score"] = np.log10(scores_agg["score"] + 1)
    return scores_agg


def create_conf_matrix(scores_df):
    categories = pd.Index(
        sorted(set(scores_df["target_label_x"]).union(set(scores_df["target_label_y"])))
    )
    cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    n = len(categories)

    conf_matrix = np.zeros((n, n))
    for row in scores_df.itertuples():
        x_idx = cat_to_idx[row.target_label_x]
        y_idx = cat_to_idx[row.target_label_y]
        conf_matrix[x_idx, y_idx] = row.score
        if x_idx != y_idx:
            conf_matrix[y_idx, x_idx] = row.score

    return conf_matrix


def plot(matrices, agg_type, model, zeroshot_dataset, seed):
    output_path = f"{zeroshot_dataset}_{model}_{agg_type}{seed}"
    model = "Sentence Transformers" if model == "sentence_transformers" else "GLiNER"
    zeroshot_dataset = display_eval.get(zeroshot_dataset, zeroshot_dataset)

    vmax = 0
    vmin = 1000
    for matrix in matrices:
        vmin = min(vmin, np.min(matrix[0][np.nonzero(matrix[0])]))
        vmax = max(vmax, np.max(matrix[0]))

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for i in range(6):
        sns.heatmap(
            matrices[i][0],
            ax=axes[i // 3, i % 3],
            cmap="magma_r",
            vmin=vmin,
            vmax=vmax,
            xticklabels=False,
            yticklabels=False,
        )
        axes[i // 3, i % 3].set_title(f"{matrices[i][1]}")
    fig.suptitle("Label Seperation Matrix using " + model + " on " + zeroshot_dataset)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/plots/{output_path}.png")
    plt.clf()


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    train_dataset_stats = get_train_datasets_stats()
    zeroshot_results = get_zeroshot_results()

    compute_sentence_transformers(train_dataset_stats, zeroshot_results)
    compute_gliner(train_dataset_stats, zeroshot_results)

    train_datasets = zeroshot_results["train_dataset"].unique()
    zeroshot_datasets = zeroshot_results["eval_benchmark"].unique()
    plot_confusion_matrices(train_datasets, zeroshot_datasets)
