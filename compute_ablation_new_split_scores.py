from analysis.data import display_train, display_eval, get_eval_scores_ablation
import pandas as pd
import json
import numpy as np
import os
from typing import List
from analysis.embedding_models import LabelEmbeddingModel, get_device, load_model
from compute_new_splits import create_splits
import torch
from tqdm import tqdm
import glob
import random
from gliner.modules.run_evaluation import create_dataset
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter


def batchify(lst, batch_size):
    return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]


def compute_embeddings(
    model: LabelEmbeddingModel, train_labels, eval_labels, batch_size: int = 32
):
    labels = list(set(train_labels + eval_labels))
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


def compute_similarities(df, train_labels, eval_labels):
    distinct_train_labels = list(set(train_labels))
    embeddings1 = df[df["label"].isin(eval_labels)]["embedding"].values
    embeddings2 = df[df["label"].isin(distinct_train_labels)]["embedding"].values

    # Step 2: Calculate cosine similarity between each pair of embeddings
    # Convert embeddings1 and embeddings2 to 2D arrays for cosine_similarity
    embeddings1 = np.stack(embeddings1)
    embeddings2 = np.stack(embeddings2)

    # Step 3: Compute cosine similarity between all pairs from list1 and list2
    similarities = cosine_similarity(embeddings1, embeddings2)
    similarities = np.clip(similarities, 0, None)

    # Convert to a DataFrame for easier interpretation
    similarity_df = pd.DataFrame(
        similarities,
        index=eval_labels,
        columns=distinct_train_labels,
    )

    return similarity_df


def compute_weighted_average(similarity_df, train_labels):
    counter = Counter(train_labels)
    weighted_scores = []
    k = 1000
    for index, row in similarity_df.iterrows():
        # Sort the similarity values in the row in descending order
        sorted_row = row.sort_values(ascending=False)
        repeated_values = np.concatenate(
            [
                np.full(counter[key], value)
                for key, value in sorted_row.items()
                if key in counter
            ]
        )[:k]

        n = len(repeated_values)
        zipf_weights = np.array([1 / k for k in range(1, n + 1)])
        zipf_weights /= zipf_weights.sum()
        weighted_sim_score = np.dot(repeated_values, zipf_weights)
        weighted_scores.append(weighted_sim_score)

    return np.mean(weighted_scores)


def compute_familarity(
    model: LabelEmbeddingModel, train_labels: List[str], eval_labels: List[str]
):
    embedding_df = compute_embeddings(model, train_labels, eval_labels)
    similarity_df = compute_similarities(embedding_df, train_labels, eval_labels)
    familarity = compute_weighted_average(similarity_df, train_labels)

    return familarity


def compute_familarity_scores():
    if os.path.exists("analysis/familarity_scores_new_splits.pkl"):
        return pd.read_pickle("analysis/familarity_scores_new_splits.pkl")

    train_data_dir = "/vol/tmp/goldejon/gliner/train_datasets/{dataset}.json"
    familarity_embedder = load_model("sentence-transformers/all-mpnet-base-v2")
    eval_labels = get_all_eval_labels()
    num_steps = 10000
    train_batch_size = 8
    required_examples = num_steps * train_batch_size
    df = pd.DataFrame(
        columns=["train_dataset", "filter_by", "difficulty", "familarity"]
    )

    for dataset in ["pilener_train", "nuner_train"]:
        for filter_by in ["entropy", "max"]:
            for setting in ["easy", "medium", "hard"]:
                train_dataset_path = train_data_dir.format(dataset=dataset)
                with open(train_dataset_path, "r") as f:
                    data = json.load(f)

                data = create_splits(
                    data,
                    dataset,
                    filter_by=filter_by,
                    setting=setting,
                )

                repeats = required_examples // len(data)
                remains = required_examples - (repeats * len(data))

                if repeats == 0 and len(data) > required_examples:
                    data = random.sample(data, required_examples)
                else:
                    if repeats > 0:
                        data = data * repeats
                    if remains > 0:
                        data = data + random.sample(data, remains)

                train_labels = [
                    item[-1].lower().strip()
                    for sublist in [x["ner"] for x in data]
                    for item in sublist
                ]

                familarity = compute_familarity(
                    familarity_embedder, train_labels, eval_labels
                )
                new_row = pd.DataFrame(
                    {
                        "train_dataset": [dataset],
                        "filter_by": [filter_by],
                        "difficulty": [setting],
                        "familarity": [familarity],
                    }
                )

                # Concatenate the new row to the existing DataFrame
                df = pd.concat([df, new_row], ignore_index=True)

    if not os.path.exists("analysis/familarity_scores_new_splits.pkl"):
        df.to_pickle("analysis/familarity_scores_new_splits.pkl")

    return df


def get_all_eval_labels():
    eval_data_dir = "/vol/tmp/goldejon/gliner/eval_datasets/NER"
    zero_shot_benc = [
        "mit-movie",
        "mit-restaurant",
        "CrossNER_AI",
        "CrossNER_literature",
        "CrossNER_music",
        "CrossNER_politics",
        "CrossNER_science",
    ]

    all_paths = glob.glob(f"{eval_data_dir}/*")
    all_paths = [
        path
        for path in all_paths
        if "sample_" not in path and path.split("/")[-1] in zero_shot_benc
    ]

    all_labels = []
    for path in all_paths:
        train, dev, test, labels = create_dataset(path)
        all_labels.extend(labels)

    all_labels = [label.lower().strip() for label in all_labels]
    return list(set(all_labels))


if __name__ == "__main__":
    familarity_scores = compute_familarity_scores()
    df = get_eval_scores_ablation(base_path="/vol/tmp/goldejon/gliner")
    df = df.drop(columns=["entity", "eval_dataset"])
    df = df.groupby(["seed", "train_dataset", "filter_by", "difficulty"]).mean()
    difficulty_order = ["easy", "medium", "hard"]

    # Convert the 'difficulty' level to categorical with the custom order
    df = df.reset_index()
    df["difficulty"] = pd.Categorical(
        df["difficulty"], categories=difficulty_order, ordered=True
    )

    # Sort the DataFrame by the 'difficulty' level
    df_sorted = df.sort_values(by=["seed", "train_dataset", "filter_by", "difficulty"])

    # Set the index back to the MultiIndex form
    df_sorted = df_sorted.set_index(
        ["seed", "train_dataset", "filter_by", "difficulty"]
    )

    output = df_sorted.reset_index().drop(columns=["seed", "precision", "recall"])
    output = output.groupby(["train_dataset", "filter_by", "difficulty"]).mean()
    output["f_score"] = output["f_score"].map(lambda x: f"{x:.3f}")
    familarity_scores = familarity_scores.groupby(
        ["train_dataset", "filter_by", "difficulty"]
    ).mean()
    familarity_scores["familarity"] = familarity_scores["familarity"].map(
        lambda x: f"{x:.3f}"
    )
    output = output.join(familarity_scores)
    output = output.rename(
        columns={
            "f_score": "F1",
            "familarity": "Familarity",
        },
    )
    output.index = output.index.set_names(["Trained On:", "Filter By", "Difficulty"])
    output = output[["Familarity", "F1"]]
    print(output.to_latex())
