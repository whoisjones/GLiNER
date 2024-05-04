import glob
import json
import os
import copy

import torch
import pandas as pd
from tqdm import tqdm
import random


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


def get_for_all_path(model, log_dir, data_paths):
    zero_shot_benc = [
        "mit-movie",
        "mit-restaurant",
        "CrossNER_AI",
        "CrossNER_literature",
        "CrossNER_music",
        "CrossNER_politics",
        "CrossNER_science",
    ]

    all_paths = glob.glob(f"{data_paths}/*")
    all_paths = [
        path
        for path in all_paths
        if "sample_" not in path and path.split("/")[-1] in zero_shot_benc
    ]

    device = next(model.parameters()).device
    model.to(device)
    model.eval()

    save_path = os.path.join(log_dir, "results.pkl")

    results = pd.DataFrame()

    for p in tqdm(all_paths):
        metrics = get_for_one_path(p, model)
        results = pd.concat([results, metrics], ignore_index=True)

    results.to_pickle(save_path)


@torch.no_grad()
def get_for_one_path(path, model):
    # load the dataset
    _, _, test_dataset, entity_types = create_dataset(path)

    data_name = path.split("/")[-1]  # get the name of the dataset

    # check if the dataset is flat_ner
    flat_ner = True
    if any([i in data_name for i in ["ACE", "GENIA", "Corpus"]]):
        flat_ner = False

    # evaluate the model
    metrics = model.evaluate(
        test_dataset,
        flat_ner=flat_ner,
        threshold=0.5,
        batch_size=12,
        entity_types=entity_types,
    )

    metrics["eval_benchmark"] = data_name

    return metrics


def get_for_all_path_with_synonyms(model, data_paths, synonyms):
    zero_shot_benc = [
        "mit-movie",
        "mit-restaurant",
        "CrossNER_AI",
        "CrossNER_literature",
        "CrossNER_music",
        "CrossNER_politics",
        "CrossNER_science",
    ]

    all_paths = glob.glob(f"{data_paths}/*")
    all_paths = [
        path
        for path in all_paths
        if "sample_" not in path and path.split("/")[-1] in zero_shot_benc
    ]

    device = next(model.parameters()).device
    model.to(device)
    model.eval()

    results = pd.DataFrame()

    for p in tqdm(all_paths):
        metrics = get_for_one_path_with_synonyms(p, model, synonyms)
        results = pd.concat([results, metrics], ignore_index=True)

    return results


@torch.no_grad()
def get_for_one_path_with_synonyms(path, model, synonyms):
    # load the dataset
    _, _, test_dataset, entity_types = create_dataset(path)

    test_datasets_iter, entity_types_iter = inject_synonyms(
        test_dataset, entity_types, synonyms
    )

    data_name = path.split("/")[-1]  # get the name of the dataset

    # check if the dataset is flat_ner
    flat_ner = True
    if any([i in data_name for i in ["ACE", "GENIA", "Corpus"]]):
        flat_ner = False

    all_results = []
    for test_dataset, entity_types in zip(test_datasets_iter, entity_types_iter):
        # evaluate the model
        metrics = model.evaluate(
            test_dataset,
            flat_ner=flat_ner,
            threshold=0.5,
            batch_size=12,
            entity_types=entity_types,
        )
        metrics["eval_benchmark"] = data_name
        all_results.append(metrics)

    metrics = transform_synonym_metrics(all_results, synonyms)

    return metrics


def inject_synonyms(test_dataset, entity_types, synonyms):
    entity_types_with_synonyms = []
    test_dataset_with_synonyms = []
    synoynms_for_replacement = list(zip(*synonyms.values()))
    original_entity_types = [list(synonyms.keys())] * len(synoynms_for_replacement)
    for original_types, _synonyms in zip(
        original_entity_types, synoynms_for_replacement
    ):
        test_dataset_with_synonyms.append(copy.deepcopy(test_dataset))
        entity_types_with_synonyms.append(copy.deepcopy(entity_types))
        for original_type, synonym in zip(original_types, _synonyms):
            if original_type in entity_types:
                entity_idx = entity_types.index(original_type)
                entity_types_with_synonyms[-1][entity_idx] = synonym
                for dp_idx, data_point in enumerate(test_dataset_with_synonyms[-1]):
                    for y_idx, entity in enumerate(data_point["ner"]):
                        if entity[2] == original_type:
                            test_dataset_with_synonyms[-1][dp_idx]["ner"][y_idx] = (
                                entity[0],
                                entity[1],
                                synonym,
                            )
    entity_types_iter = entity_types_with_synonyms
    test_dataset_iter = test_dataset_with_synonyms

    return test_dataset_iter, entity_types_iter


def transform_synonym_metrics(metrics_list, synonyms):
    metrics = {
        "original_label": [],
        "synonym": [],
        "metric": [],
        "value": [],
        "is_synonym": [],
        "eval_benchmark": [],
    }

    synoynms_for_replacement = list(zip(*synonyms.values()))
    original_entity_types = [list(synonyms.keys())] * len(synoynms_for_replacement)

    for idx, (original_types, _synonyms) in enumerate(
        zip(original_entity_types, synoynms_for_replacement)
    ):
        for original_type, synonym in zip(original_types, _synonyms):
            if synonym not in metrics_list[idx]["entity"].values:
                continue
            for metric in ["precision", "recall", "f_score"]:
                metrics["original_label"].append(original_type)
                metrics["synonym"].append(synonym)
                metrics["value"].append(
                    metrics_list[idx][metrics_list[idx]["entity"] == synonym][
                        metric
                    ].values.item()
                )
                metrics["metric"].append(metric)
                metrics["eval_benchmark"].append(
                    metrics_list[idx][metrics_list[idx]["entity"] == synonym][
                        "eval_benchmark"
                    ].values.item()
                )
                metrics["is_synonym"].append(
                    True if original_type != synonym else False
                )

    metrics = pd.DataFrame.from_dict(metrics)
    return metrics


def sample_train_data(data_paths, sample_size=10000):
    all_paths = glob.glob(f"{data_paths}/*")

    all_paths = sorted(all_paths)

    # to exclude the zero-shot benchmark datasets
    zero_shot_benc = [
        "CrossNER_AI",
        "CrossNER_literature",
        "CrossNER_music",
        "CrossNER_politics",
        "CrossNER_science",
        "ACE 2004",
    ]

    new_train = []
    # take 10k samples from each dataset
    for p in tqdm(all_paths):
        if any([i in p for i in zero_shot_benc]):
            continue
        train, dev, test, labels = create_dataset(p)

        # add label key to the train data
        for i in range(len(train)):
            train[i]["label"] = labels

        random.shuffle(train)
        train = train[:sample_size]
        new_train.extend(train)

    return new_train
