import glob
import json
import os
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
    "asknews": "AskNews",
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


def get_train_datasets_stats(
    train_datasets: list = [
        "neretrieve_train",
        "litset",
        "asknews",
        "pilener_train",
        "nuner_train",
        "ontonotes",
        "fewnerd",
    ],
    base_path: str = "/vol/tmp/goldejon/gliner",
):
    path = f"{base_path}/analysis/full_train_statistics.pkl"
    if os.path.exists(path):
        return pd.read_pickle(path)

    train_datasets_stats = pd.DataFrame()
    for dataset_name in train_datasets:

        print(f"Loading in {dataset_name}.")
        if dataset_name == "litset":
            with open(f"{base_path}/train_datasets/litset.jsonl", "r") as f:
                data = []
                for line in f.readlines():
                    data.append(json.loads(line))
                dataset = [x for x in data if x["ner"]]
        elif dataset_name == "asknews":
            with open(f"{base_path}/train_datasets/asknews.json", "r") as f:
                asknews = json.load(f)
            with open(f"{base_path}/train_datasets/pilener_train.json", "r") as f:
                pilener = json.load(f)
            dataset = asknews + pilener
        else:
            with open(f"{base_path}/train_datasets/{dataset_name}.json", "r") as f:
                dataset = json.load(f)

        len_train_dataset = len(dataset)
        print(f"Number of sentences in {dataset_name}: {len_train_dataset}")

        print("Sampling labels.")
        if dataset_name == "asknews":
            labels_pilener = get_train_labels(
                pilener, "pilener_train", base_path=base_path
            )
            labels_asknews = get_train_labels(asknews, "asknews", base_path=base_path)
            train_labels = labels_pilener + labels_asknews
        else:
            train_labels = get_train_labels(dataset, dataset_name, base_path=base_path)

        required_examples = 240000
        repeats = required_examples // len(dataset)
        remains = required_examples - (repeats * len(dataset))

        if repeats == 0 and len(train_labels) > required_examples:
            sampled_labels = random.sample(train_labels, required_examples)
        else:
            if repeats > 0:
                sampled_labels = train_labels * repeats
            if remains > 0:
                sampled_labels = train_labels + random.sample(train_labels, remains)

        train_labels = [item for sublist in train_labels for item in sublist]
        sampled_labels = [item for sublist in sampled_labels for item in sublist]

        train_labels_full_dataset = [label.lower() for label in train_labels]
        train_labels_set_full_dataset = set(train_labels_full_dataset)
        train_labels_counter_full_dataset = Counter(train_labels)

        train_labels_sampled = [label.lower() for label in sampled_labels]
        train_labels_set_sampled = set(train_labels_sampled)
        train_labels_counter_sampled = Counter(train_labels_sampled)

        df = pd.DataFrame(
            {
                "train_dataset": [dataset_name],
                "num_sentences": [len_train_dataset],
                "train_labels_set_sampled": [train_labels_set_sampled],
                "train_labels_counter_sampled": [train_labels_counter_sampled],
                "train_labels_set_full_dataset": [train_labels_set_full_dataset],
                "train_labels_counter_full_dataset": [
                    train_labels_counter_full_dataset
                ],
            }
        )

        train_datasets_stats = pd.concat([train_datasets_stats, df])

    train_datasets_stats.reset_index(drop=True, inplace=True)

    train_datasets_stats.to_pickle(path)

    return train_datasets_stats


def get_train_labels(
    dataset,
    dataset_name: str,
    base_path: str = "/vol/tmp/goldejon/gliner",
):
    train_datasets_path = f"{base_path}/train_datasets"
    if dataset_name == "litset":
        with open(f"{train_datasets_path}/litset_labels.json", "r") as f:
            id2label = json.load(f)
        id2label.pop("0")

        labels = []
        for dp in dataset:
            sentence_labels = []
            for span in dp["ner"]:
                label_type = random.choice(["description", "labels"])
                fallback_type = "description" if label_type == "labels" else "labels"
                annotation = span[-1]

                if not annotation[label_type]:
                    continue

                if label_type in annotation:
                    sentence_labels.append(
                        random.choice(annotation[label_type])
                        if label_type == "labels"
                        else annotation[label_type][0]
                    )
                elif fallback_type in annotation:
                    sentence_labels.append(
                        random.choice(annotation[fallback_type])
                        if fallback_type == "labels"
                        else annotation[fallback_type][0]
                    )
                else:
                    sentence_labels.append("miscellaneous")

            labels.append(sentence_labels)

    else:
        labels = []
        for dp in dataset:
            labels.append(
                [
                    (
                        random.choice(entity[-1])
                        if dataset_name == "neretrieve_train"
                        else entity[-1]
                    )
                    for entity in dp["ner"]
                ]
            )

    return labels


def get_eval_datasets_stats(
    base_path: str = "/vol/tmp/goldejon/gliner",
):
    stats_path = f"{base_path}/analysis/eval_statistics.pkl"
    eval_datasets_path = f"{base_path}/eval_datasets/NER"
    if os.path.exists(stats_path):
        return pd.read_pickle(stats_path)

    eval_datasets = [
        "mit-movie",
        "mit-restaurant",
        "CrossNER_AI",
        "CrossNER_literature",
        "CrossNER_music",
        "CrossNER_politics",
        "CrossNER_science",
    ]
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
                "num_sentences": [len(dataset)],
                "eval_labels_set": [eval_labels_binary],
                "eval_labels_counter": [eval_labels_count],
            }
        )

        eval_datasets_stats = pd.concat([eval_datasets_stats, df])

    eval_datasets_stats.reset_index(drop=True, inplace=True)

    eval_datasets_stats.to_pickle(stats_path)

    return eval_datasets_stats


def compute_overlaps(row):
    exact_binary_counter = 0
    exact_example_sum = 0
    exact_example_counter = Counter()

    exact_matches = set(
        [tl for tl in row["train_labels_set_sampled"] if tl in row["eval_labels_set"]]
    )
    for exact_match in exact_matches:
        exact_binary_counter += 1
        num_examples_seen_for_label = row["train_labels_counter_sampled"][exact_match]
        exact_example_sum += num_examples_seen_for_label
        exact_example_counter.update({exact_match: num_examples_seen_for_label})

    substring_example_sum = 0
    substring_example_counter = Counter()
    substring_matches = set()
    for el in row["eval_labels_set"]:
        for tl in row["train_labels_set_sampled"]:
            if el in tl:
                num_examples_seen_for_label = row["train_labels_counter_sampled"][tl]
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


def get_eval_scores(base_path: str = "/vol/tmp/goldejon/gliner"):
    paths = f"{base_path}/eval/*/*/results.pkl"
    paths = glob.glob(paths)

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


def get_eval_scores_ablation(base_path: str = "/vol/tmp/goldejon/gliner"):
    paths = f"{base_path}/logs_ablation_new_splits/*/*/*/*/results.pkl"
    paths = glob.glob(paths)

    all_results = pd.DataFrame()
    for path in paths:
        result = pd.read_pickle(path)
        metadata = path.split("/")
        dataset = metadata[-5]
        difficulty = metadata[-4]
        filter_by = metadata[-3]
        seed = metadata[-2]
        result["train_dataset"] = dataset
        result["difficulty"] = difficulty
        result["filter_by"] = filter_by
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


def detailed_exposure_type_during_training(row):
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


def exposure_type_during_training(row):
    exact_seen = row["exact_overlap_counter"][row["entity"]]
    exact_and_partially_seen = row["partial_overlap_counter"][row["entity"]]
    if exact_and_partially_seen > exact_seen:
        seen_during_training = "exact + substring matches"
    elif exact_seen > 0:
        seen_during_training = "exact match"
    elif exact_and_partially_seen == 0:
        seen_during_training = "true zero-shot"
    else:
        raise ValueError("This should not happen")

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
