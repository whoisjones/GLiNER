import json

import torch
from scipy.stats import entropy
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

from gliner.modules.run_evaluation import create_dataset


def load_train_dataset(dataset_name):
    with open("/home/ec2-user/paper_data/train_datasets/" + dataset_name, "r") as f:
        data = json.load(f)
    return data


def main():
    train_datasets = ["pilener_train.json"]  # , "nuner_train.json"]
    benchmark_names = [
        "mit-movie",
        "mit-restaurant",
        "CrossNER_AI",
        "CrossNER_literature",
        "CrossNER_music",
        "CrossNER_politics",
        "CrossNER_science",
    ]

    benchmarks = {}
    for benchmark_name in benchmark_names:
        _, _, test_dataset, entity_types = create_dataset(
            "/home/ec2-user/paper_data/eval_datasets/NER/" + benchmark_name
        )
        benchmarks[benchmark_name] = entity_types

    training_datasets = {}
    for train_dataset_name in train_datasets:
        train_dataset = load_train_dataset(train_dataset_name)
        entity_types = set()
        for dp in train_dataset:
            annotations = [x[-1] for x in dp["ner"]]
            entity_types.update(annotations)
        training_datasets[train_dataset_name] = list(entity_types)

    rankings = {}
    batch_size = 256
    model = SentenceTransformer("all-mpnet-base-v2").to("cuda")
    eval_encodings = {}
    for benchmark_name, entity_types in benchmarks.items():
        embeddings = model.encode(entity_types, convert_to_tensor=True, device="cuda")
        eval_encodings[benchmark_name] = embeddings

    results = {}
    for dataset_name, entity_types in training_datasets.items():
        for i in tqdm(range(0, len(entity_types), batch_size)):
            batch = entity_types[i : i + batch_size]
            embeddings = model.encode(batch, convert_to_tensor=True, device="cuda")
            for benchmark_name, eval_embeddings in eval_encodings.items():
                similarities = cosine_similarity(
                    embeddings.unsqueeze(1),
                    eval_embeddings.unsqueeze(0),
                    dim=2,
                )
                probabilities = torch.nn.functional.softmax(similarities / 0.01, dim=1)
                entropy_values = -torch.sum(
                    probabilities * torch.log(probabilities + 1e-10), dim=1
                )

                if dataset_name not in results:
                    results[dataset_name] = {}
                if benchmark_name not in results[dataset_name]:
                    results[dataset_name][benchmark_name] = {}

                for j, entity in enumerate(batch):
                    results[dataset_name][benchmark_name][entity] = (
                        entropy_values[j].cpu().numpy().item()
                    )

    for dataset_name, eval_comparisons in results.items():
        for benchmark_name, mapping in eval_comparisons.items():
            out = dict(sorted(mapping.items(), key=lambda x: x[1]))
            print()


if __name__ == "__main__":
    main()
