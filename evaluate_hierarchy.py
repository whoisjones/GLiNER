import os
import json
import random
import copy

import torch
import pandas as pd

from gliner import GLiNER

from evaluate import configs

hierarchy = {
    "writer": [
        "indian writer",
        "british writer",
        "polish writer",
        "norwegian writer",
        "north american writer",
    ],
    "film": [
        "comedy film",
        "horror film",
        "animated film",
        "science fiction film",
        "musical film",
    ],
    "athlete": ["cyclist", "boxer", "swimmer", "basketball player", "climber"],
    "professional": [
        "mathematician",
        "economist",
        "botanist",
        "neuroscientist",
        "space scientist",
    ],
}


def filter_neretrieve(dataset, coarse_type):
    filtered_dataset = []
    for sample in dataset:
        targets = set([i for j in [d[-1] for d in sample["ner"]] for i in j])
        if any([x in targets for x in hierarchy[coarse_type]]):
            filtered_dataset.append(sample)
    return filtered_dataset


def relabel(coarse_subset, fine_subset, coarse_type):

    for idx, sample in enumerate(coarse_subset):
        coarse_labels = []
        fine_labels = []
        for label in sample["ner"]:
            matches = [y for y in hierarchy[coarse_type] if y in label[2]]
            if any(matches):
                coarse_labels.append([label[0], label[1], coarse_type])
                fine_labels.append([label[0], label[1], random.choice(matches)])

        coarse_subset[idx]["ner"] = coarse_labels
        fine_subset[idx]["ner"] = fine_labels

    return coarse_subset, fine_subset


def evaluate(test_dataset, entity_types, coarse_type, coarse):
    results = pd.DataFrame()
    for dataset, model_paths in configs.items():
        for model_path in model_paths:
            model_ckpt = (
                f"/vol/tmp/goldejon/gliner/logs/deberta-v3-large/{dataset}/{model_path}"
            )

            model = GLiNER.from_pretrained(model_ckpt)
            model.eval()
            if torch.cuda.is_available():
                model = model.to("cuda")

            metrics = model.evaluate(
                test_dataset,
                flat_ner=True,
                threshold=0.5,
                batch_size=12,
                entity_types=entity_types,
            )

            metrics = pd.DataFrame.from_dict(metrics)
            metrics["train_dataset"] = dataset
            results = pd.concat([results, metrics], ignore_index=True)

    return results


def main():
    with open(
        "/vol/tmp/goldejon/gliner/train_datasets/neretrieve_train.json", "r"
    ) as f:
        data = json.load(f)

    results = pd.DataFrame()

    output_dir = f"/vol/tmp/goldejon/gliner/eval_hierarchy"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for coarse_type, fine_types in hierarchy.items():
        filtered_dataset = filter_neretrieve(data, coarse_type)
        subset = random.sample(filtered_dataset, 15000)

        coarse_subset = copy.deepcopy(subset)
        fine_subset = copy.deepcopy(subset)

        coarse_subset, fine_subset = relabel(coarse_subset, fine_subset, coarse_type)
        corase_results = evaluate(
            coarse_subset, [coarse_type], coarse_type=coarse_type, coarse=True
        )
        fine_results = evaluate(
            fine_subset, fine_types, coarse_type=coarse_type, coarse=False
        )

        results = pd.concat([results, corase_results, fine_results], ignore_index=True)

    results.to_pickle(f"{output_dir}/results.pkl")


if __name__ == "__main__":
    main()
