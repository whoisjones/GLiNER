import os
import json
import glob
import random
from collections import Counter

import pandas as pd

display_train = {
    "ontonotes": "OntoNotes",
    "fewnerd": "FewNERD",
    "neretrieve_train": "NERetrieve",
    "litset": "LitSet",
    "nuner_train": "NuNER",
    "pilener_train": "PileNER",
}

display_eval = {
    "mit-movie": "Movie",
    "mit-restaurant": "Restaurant",
    "CrossNER_AI": "AI",
    "CrossNER_science": "Science",
    "CrossNER_politics": "Politics",
    "CrossNER_literature": "Literature",
    "CrossNER_music": "Music",
}


def get_train_datasets_stats():
    if os.path.exists("/vol/tmp/goldejon/gliner/analysis/train_statistics.pkl"):
        return pd.read_pickle("/vol/tmp/goldejon/gliner/analysis/train_statistics.pkl")

    train_datasets = [
        "ontonotes",
        "fewnerd",
        "litset",
        "nuner_train",
        "pilener_train",
        "neretrieve_train",
    ]
    train_datasets_stats = pd.DataFrame()
    for train_dataset in train_datasets:
        train_labels = get_train_labels(train_dataset)
        train_labels = [label.lower() for label in train_labels]

        train_labels_binary = set(train_labels)
        train_labels_count = Counter(train_labels)

        df = pd.DataFrame(
            {
                "train_dataset": [train_dataset],
                "train_labels_set": [train_labels_binary],
                "train_labels_counter": [train_labels_count],
            }
        )

        train_datasets_stats = pd.concat([train_datasets_stats, df])

    train_datasets_stats.reset_index(drop=True, inplace=True)

    train_datasets_stats.to_pickle(
        "/vol/tmp/goldejon/gliner/analysis/train_statistics.pkl"
    )

    return train_datasets_stats


def get_train_labels(
    train_dataset: str,
    required_examples: int = 240000,
):
    train_datasets_path: str = "/vol/tmp/goldejon/gliner/train_datasets"
    if train_dataset == "litset":
        with open("/vol/tmp/goldejon/ner4all/loner/labelID2label.json", "r") as f:
            id2label = json.load(f)
        id2label.pop("0")

        data = random.sample([label for label in id2label.values()], required_examples)

        labels = []
        for label in data:
            label_type = random.choice(["description", "labels"])
            fallback_type = "description" if label_type == "labels" else "labels"
            if label_type in label:
                labels.append(
                    random.choice(label[label_type])
                    if label_type == "labels"
                    else label[label_type]
                )
            elif fallback_type in label:
                labels.append(
                    random.choice(label[fallback_type])
                    if fallback_type == "labels"
                    else label[fallback_type]
                )
            else:
                labels.append("miscellaneous")
    elif train_dataset == "neretrieve_train":
        with open(os.path.join(train_datasets_path, f"{train_dataset}.json"), "r") as f:
            data = json.load(f)

        data = random.sample(data, required_examples)

        labels = []
        for dp in data:
            for entity in dp["ner"]:
                labels.append(random.choice(entity[-1]))

    else:
        with open(os.path.join(train_datasets_path, f"{train_dataset}.json"), "r") as f:
            data = json.load(f)

        repeats = required_examples // len(data)
        remains = required_examples - (repeats * len(data))

        if repeats > 0:
            data = data * repeats
        if remains > 0:
            data = data + random.sample(data, remains)

        labels = []
        for dp in data:
            for entity in dp["ner"]:
                labels.append(entity[-1])

    return labels


def get_eval_datasets_stats():
    if os.path.exists("/vol/tmp/goldejon/gliner/analysis/eval_statistics.pkl"):
        return pd.read_pickle("/vol/tmp/goldejon/gliner/analysis/eval_statistics.pkl")

    eval_datasets = [
        "mit-movie",
        "mit-restaurant",
        "CrossNER_AI",
        "CrossNER_literature",
        "CrossNER_music",
        "CrossNER_politics",
        "CrossNER_science",
    ]
    eval_datasets_path: str = "/vol/tmp/goldejon/gliner/eval_datasets/NER"
    eval_datasets_stats = pd.DataFrame()
    for eval_dataset in eval_datasets:
        _, _, dataset, _ = create_dataset(
            os.path.join(eval_datasets_path, eval_dataset)
        )

        eval_labels = []
        for dp in dataset:
            for entity in dp["ner"]:
                eval_labels.append(entity[-1].lower())

        eval_labels_binary = set(eval_labels)
        eval_labels_count = Counter(eval_labels)

        df = pd.DataFrame(
            {
                "eval_dataset": [eval_dataset],
                "eval_labels_set": [eval_labels_binary],
                "eval_labels_counter": [eval_labels_count],
            }
        )

        eval_datasets_stats = pd.concat([eval_datasets_stats, df])

    eval_datasets_stats.reset_index(drop=True, inplace=True)

    eval_datasets_stats.to_pickle(
        "/vol/tmp/goldejon/gliner/analysis/eval_statistics.pkl"
    )

    return eval_datasets_stats


def compute_overlaps(row):
    exact_binary_counter = 0
    exact_example_sum = 0
    exact_example_counter = Counter()

    exact_matches = set(
        [tl for tl in row["train_labels_set"] if tl in row["eval_labels_set"]]
    )
    for exact_match in exact_matches:
        exact_binary_counter += 1
        num_examples_seen_for_label = row["train_labels_counter"][exact_match]
        exact_example_sum += num_examples_seen_for_label
        exact_example_counter.update({exact_match: num_examples_seen_for_label})

    substring_example_sum = 0
    substring_example_counter = Counter()
    substring_matches = set()
    for el in row["eval_labels_set"]:
        for tl in row["train_labels_set"]:
            if el in tl:
                num_examples_seen_for_label = row["train_labels_counter"][tl]
                substring_matches.add(el)
                substring_example_sum += num_examples_seen_for_label
                substring_example_counter.update({el: num_examples_seen_for_label})

    exact_overlap = exact_binary_counter / len(row["eval_labels_set"])
    partial_overlap = len(substring_matches) / len(row["eval_labels_set"])

    row["exact_overlap"] = exact_matches
    row["partial_overlap"] = substring_matches
    row["exact_overlap_percentage"] = exact_overlap
    row["partial_overlap_percentage"] = partial_overlap
    row["exact_overlap_sum"] = exact_example_sum
    row["partial_overlap_sum"] = substring_example_sum
    row["exact_overlap_counter"] = exact_example_counter
    row["partial_overlap_counter"] = substring_example_counter

    return row


def get_eval_scores():
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

    all_results = all_results[all_results["entity"] != "average"]
    all_results["entity"] = all_results["entity"].str.lower()
    all_results.rename(columns={"eval_benchmark": "eval_dataset"}, inplace=True)
    all_results.reset_index(drop=True, inplace=True)

    return all_results


def bin_eval_labels(scores):
    scores = scores.apply(create_raw_bin, axis=1)
    bins = [0, 50, 100, 250, 500, float("inf")]
    labels = ["0-50", "50-100", "100-250", "250-500", "500+"]
    scores["eval_label_bin"] = pd.cut(
        scores["eval_label_bin"], bins, labels=labels, include_lowest=True
    )
    return scores


def create_raw_bin(row):
    row["eval_label_bin"] = row["eval_labels_counter"].get(row["entity"], 0)
    return row


def exposure_type_during_training(row):
    exact_seen = row["exact_overlap_counter"][row["entity"]]
    exact_and_partially_seen = row["partial_overlap_counter"][row["entity"]]
    partially_seen = exact_and_partially_seen - exact_seen
    if exact_seen and partially_seen:
        seen_during_training = "exact + partial"
    elif exact_seen:
        seen_during_training = "exact only"
    elif partially_seen:
        seen_during_training = "partial only"
    else:
        seen_during_training = "not seen"

    row["exposure_type"] = seen_during_training

    return row


def count_label_seen(row):
    row["times_entity_seen_exact"] = row["exact_overlap_counter"].get(row["entity"], 0)
    row["times_entity_seen_partial"] = row["partial_overlap_counter"].get(
        row["entity"], 0
    )
    return row


def open_content(path):
    paths = glob.glob(os.path.join(path, "*.json"))
    train, dev, test, labels = None, None, None, None
    for p in paths:
        if "train" in p:
            with open(p, "r") as f:
                train = json.load(f)
        elif "dev" in p:
            with open(p, "r") as f:
                dev = json.load(f)
        elif "test" in p:
            with open(p, "r") as f:
                test = json.load(f)
        elif "labels" in p:
            with open(p, "r") as f:
                labels = json.load(f)
    return train, dev, test, labels


def process(data):
    words = data["sentence"].split()
    entities = []  # List of entities (start, end, type)

    for entity in data["entities"]:
        start_char, end_char = entity["pos"]

        # Initialize variables to keep track of word positions
        start_word = None
        end_word = None

        # Iterate through words and find the word positions
        char_count = 0
        for i, word in enumerate(words):
            word_length = len(word)
            if char_count == start_char:
                start_word = i
            if char_count + word_length == end_char:
                end_word = i
                break
            char_count += word_length + 1  # Add 1 for the space

        # Append the word positions to the list
        entities.append((start_word, end_word, entity["type"]))

    # Create a list of word positions for each entity
    sample = {"tokenized_text": words, "ner": entities}

    return sample


# create dataset
def create_dataset(path):
    train, dev, test, labels = open_content(path)
    train_dataset = []
    dev_dataset = []
    test_dataset = []
    for data in train:
        train_dataset.append(process(data))
    for data in dev:
        dev_dataset.append(process(data))
    for data in test:
        test_dataset.append(process(data))
    return train_dataset, dev_dataset, test_dataset, labels
