import os
import copy
import json
import random
import logging
import pandas as pd

from save_load import load_model

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create formatter with timestamp
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Create console handler and set level to debug
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Add formatter to console handler
console_handler.setFormatter(formatter)

# Add console handler to logger
logger.addHandler(console_handler)

labels = {
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
    "profession": [
        "mathematician",
        "economist",
        "botanist",
        "neuroscientist",
        "space scientist",
    ],
}


def filter_neretrieve(dataset, coarse_type):
    filtered = []
    for sample in dataset:
        targets = set([i for j in [d[-1] for d in sample["ner"]] for i in j])
        if any([x in targets for x in labels[coarse_type]]):
            filtered.append(sample)
    return filtered


def relabel(coarse_subset, fine_subset, coarse_type):

    for idx, sample in enumerate(coarse_subset):
        coarse_labels = []
        fine_labels = []
        for label in sample["ner"]:
            matches = [y for y in labels[coarse_type] if y in label[2]]
            if any(matches):
                coarse_labels.append([label[0], label[1], coarse_type])
                fine_labels.append([label[0], label[1], random.choice(matches)])

        coarse_subset[idx]["ner"] = coarse_labels
        fine_subset[idx]["ner"] = fine_labels

    return coarse_subset, fine_subset


def evaluate(test_dataset, entity_types, coarse_type, coarse):
    configs = {
        "pilener_train": ["2024-03-12_14-40-13/model_30000"],
        "ontonotes": ["2024-03-12_17-05-12/model_30000"],
        "litset": ["2024-03-13_16-26-19/model_30000"],
        "fewnerd": ["2024-03-13_09-53-49/model_30000"],
    }
    for dataset, model_paths in configs.items():
        for model_path in model_paths:
            model_ckpt = (
                f"/vol/tmp/goldejon/gliner/logs/deberta-v3-small/{dataset}/{model_path}"
            )

            output_dir = f"/vol/tmp/goldejon/gliner/eval_intersections/{dataset}/{model_path.split('/')[0]}"

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            model = load_model(model_ckpt)
            device = next(model.parameters()).device
            model.to(device)
            model.eval()

            metrics = model.evaluate(
                test_dataset,
                flat_ner=True,
                threshold=0.5,
                batch_size=12,
                entity_types=entity_types,
            )

            results = pd.DataFrame.from_dict(metrics)
            results.to_pickle(
                os.path.join(
                    output_dir, f"{coarse_type}_{'coarse' if coarse else 'fine'}.pkl"
                )
            )


def main():

    # Load dataset
    logger.info("Loading dataset.")
    with open(
        "/vol/tmp/goldejon/gliner/train_datasets/neretrieve_train.json", "r"
    ) as f:
        data = json.load(f)

    # Sample labels from each coarse category
    for coarse_type, fine_types in labels.items():
        logger.info(f"Sampling labels for {coarse_type}.")
        filtered_dataset = filter_neretrieve(data, coarse_type)
        subset = random.sample(filtered_dataset, 15000)

        coarse_subset = copy.deepcopy(subset)
        fine_subset = copy.deepcopy(subset)

        logger.info("Relabeling samples.")
        coarse_subset, fine_subset = relabel(coarse_subset, fine_subset, coarse_type)

        logger.info("Evaluating coarse samples.")
        evaluate(coarse_subset, [coarse_type], coarse_type=coarse_type, coarse=True)
        logger.info("Evaluating fine samples.")
        evaluate(fine_subset, fine_types, coarse_type=coarse_type, coarse=False)


if __name__ == "__main__":
    main()
