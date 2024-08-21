import glob
import json
import math
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from sentence_transformers import util
from tqdm import tqdm

from data import display_train, get_eval_scores, get_train_datasets_stats


def get_device():
    """Whether to use GPU or CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_glove(glove_file):
    """
    Load GloVe model from a text file.

    Args:
        glove_file (str): Path to the GloVe text file.

    Returns:
        dict: A dictionary mapping words to their GloVe vector representations.
    """
    glove = {}
    with open(glove_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]])
            glove[word] = vector
    return glove


def compute_embeddings(train_dataset_stats, zeroshot_results):
    weights = torch.load("/vol/tmp/goldejon/glove/torch_embedding.pt")
    embedding = torch.nn.Embedding.from_pretrained(weights)

    with open("/vol/tmp/goldejon/glove/vocab.json", "r") as f:
        vocab = json.load(f)

    embedding.to(get_device())

    zeroshot_labels = zeroshot_results["entity"].unique().tolist()
    zeroshot_labels = [label.lower().split(" ") for label in zeroshot_labels]
    train_labels_df = train_dataset_stats[["train_dataset", "train_labels_set"]]
    train_labels_normalized = train_labels_df.explode("train_labels_set")
    train_labels = [
        label.lower().split(" ")
        for label in train_labels_normalized["train_labels_set"].unique().tolist()
    ]

    zeroshot_input_ids = [
        [vocab.get(label, vocab.get("<unk>")) for label in labels]
        for labels in zeroshot_labels
    ]
    train_input_ids = [
        [vocab.get(label, vocab.get("<unk>")) for label in labels]
        for labels in train_labels
    ]

    zeroshot_embeddings = []
    train_label_embeddings = []

    if not os.path.exists(f"{OUTPUT_PATH}/zeroshot_embeddings.pkl"):
        for i in tqdm(range(0, len(zeroshot_labels)), desc="Zeroshot Embedding"):
            input_ids = torch.LongTensor(zeroshot_input_ids[i]).to("cuda")
            emb = embedding(input_ids).detach().cpu().numpy()
            emb.mean(axis=0)
            zeroshot_embeddings.append(emb)

        embeddings = np.concatenate(zeroshot_embeddings)
        zeroshot_embedding_df = {"label": [], "embedding": []}
        for i in range(len(zeroshot_labels)):
            zeroshot_embedding_df["label"].append(" ".join(zeroshot_labels[i]))
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
        for i in tqdm(range(0, len(train_labels)), desc="Train Label Embedding"):
            input_ids = torch.LongTensor(train_input_ids[i]).to("cuda")
            emb = embedding(input_ids).detach().cpu().numpy()
            emb.mean(axis=0)
            train_label_embeddings.append(emb)

        embeddings = np.concatenate(train_label_embeddings)
        train_label_embedding_df = {"label": [], "embedding": []}
        for i in range(len(train_labels)):
            train_label_embedding_df["label"].append(" ".join(train_labels[i]))
            train_label_embedding_df["embedding"].append(embeddings[i])
        train_label_embedding_df = pd.DataFrame.from_dict(train_label_embedding_df)

        output_train = pd.merge(
            train_labels_normalized,
            train_label_embedding_df,
            left_on="train_labels_set",
            right_on="label",
            how="inner",
        ).drop(columns="train_labels_set")
        output_train.to_pickle(f"{OUTPUT_PATH}/train_label_embeddings.pkl")
    else:
        output_train = pd.read_pickle(f"{OUTPUT_PATH}/train_label_embeddings.pkl")

    return output_train, output_zeroshot


def compute_similarity(train_stats, train_embeddings, zeroshot_embeddings):
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
            top_k = sim.argsort()[0].tolist()[::-1][:100]

            closest_similarities = sim[0][top_k]
            closest_labels = dataset_embeddings.iloc[top_k]["label"].tolist()
            train_stats[train_stats["train_dataset"] == row["train_dataset"]][
                "train_labels_counter"
            ]
            counter = train_stats[train_stats["train_dataset"] == row["train_dataset"]][
                "train_labels_counter"
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_path", type=str, default="logs")
    parser.add_argument("--results_dir", type=str, default="/home/ec2-user/paper_data")
    parser.add_argument("--glove_files", nargs="+", type=str)
    parser.add_argument("--transformer_models", nargs="+", type=str)
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    train_statistics = get_train_datasets_stats(base_path=args.results_dir)
    scores = get_eval_scores(base_path=args.results_dir)

    embedding_paths_or_ = args.glove_files + args.transformer_models

    compute_metric(embeddings, train_statistics, scores)
