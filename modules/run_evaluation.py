import glob
import json
import os
import copy

import torch
from tqdm import tqdm
import random
import pandas as pd


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


def transform_metrics(metrics, dataset_name):
    metrics.reset_index(inplace=True)
    metrics.rename(columns={"index": "metric"}, inplace=True)
    metrics = pd.melt(
        metrics, id_vars=["metric"], var_name="entity", value_name="value"
    )
    metrics["dataset_name"] = dataset_name

    return metrics


def transform_synonym_metrics(metrics_list, dataset_name, synonyms):
    synoynms_for_replacement = list(zip(*synonyms.values()))
    original_entity_types = [list(synonyms.keys())] * len(synoynms_for_replacement)
    metrics = pd.DataFrame()
    for idx, (original_types, _synonyms) in enumerate(
        zip(original_entity_types, synoynms_for_replacement)
    ):
        for original_type, synonym in zip(original_types, _synonyms):
            if synonym not in metrics_list[idx].columns:
                continue
            scores = metrics_list[idx][synonym].reset_index()
            scores.rename(columns={"index": "metric", synonym: "value"}, inplace=True)
            scores["original_label"] = original_type
            scores["synonym_label"] = synonym
            metrics = pd.concat([metrics, scores])

    if not metrics.empty:
        metrics["is_synonym"] = metrics["original_label"] != metrics["synonym_label"]
        metrics["dataset_name"] = dataset_name

        metrics.reset_index(drop=True, inplace=True)
        return metrics
    else:
        return None


@torch.no_grad()
def get_for_one_path(path, model, dataloader=None, synonyms=None):
    # load the dataset
    _, _, test_dataset, entity_types = create_dataset(path)

    if synonyms is not None:
        test_datasets_iter, entity_types_iter = inject_synonyms(
            test_dataset, entity_types, synonyms
        )
    else:
        test_datasets_iter = [test_dataset]
        entity_types_iter = [entity_types]

    metrics_list = []
    for test_dataset, entity_types in zip(test_datasets_iter, entity_types_iter):

        data_name = path.split("/")[-1]  # get the name of the dataset

        # check if the dataset is flat_ner
        flat_ner = True
        if any([i in data_name for i in ["ACE", "GENIA", "Corpus"]]):
            flat_ner = False

        if dataloader:
            test_dataset = dataloader.get_validation_loader(
                test_dataset, batch_size=12, shuffle=False, entity_types=entity_types
            )

        # evaluate the model
        metrics = model.evaluate(
            test_dataset,
            flat_ner=flat_ner,
            threshold=0.5,
        )

        if metrics:
            metrics_list.append(
                pd.DataFrame.from_dict(metrics).apply(lambda x: x * 100).round(2)
            )

    if synonyms:
        metrics = transform_synonym_metrics(metrics_list, data_name, synonyms)
    else:
        metrics = transform_metrics(metrics_list[0], data_name)

    return metrics


def save_metrics(all_results, log_dir):
    overall_scores = all_results[all_results["entity"] == "overall"]
    for zeroshot in overall_scores["zeroshot"].unique():
        for metric in overall_scores["metric"].unique():
            avg = round(
                overall_scores[
                    (overall_scores["zeroshot"] == zeroshot)
                    & (overall_scores["metric"] == metric)
                ]["value"].mean(),
                2,
            )
            avg_row = pd.DataFrame.from_dict(
                {
                    "metric": [metric],
                    "entity": ["overall"],
                    "value": [avg],
                    "dataset_name": ["Average"],
                    "zeroshot": [zeroshot],
                }
            )
            overall_scores = pd.concat([overall_scores, avg_row], ignore_index=True)
    overall_scores.to_pickle(os.path.join(log_dir, "overall_scores.pkl"))

    scores_per_dataset = all_results[all_results["entity"] != "overall"]
    for dataset in scores_per_dataset["dataset_name"].unique():
        dataset_scores = scores_per_dataset[
            scores_per_dataset["dataset_name"] == dataset
        ]
        dataset_scores.to_pickle(os.path.join(log_dir, f"{dataset}.pkl"))


def save_metrics_synonyms(all_results, log_dir):
    all_results.to_pickle(os.path.join(log_dir, "overall_scores.pkl"))

    for dataset in all_results["dataset_name"].unique():
        dataset_scores = all_results[all_results["dataset_name"] == dataset]
        dataset_scores.to_pickle(os.path.join(log_dir, f"{dataset}.pkl"))


def get_for_all_path(
    model, log_dir, data_paths, dataloader=None, only_zero_shot=True, synonyms=None
):
    all_paths = glob.glob(f"{data_paths}/*")

    all_paths = sorted(all_paths)

    # move the model to the device
    device = next(model.parameters()).device
    model.to(device)
    # set the model to eval mode
    model.eval()

    zero_shot_benc = [
        "mit-movie",
        "mit-restaurant",
        "CrossNER_AI",
        "CrossNER_literature",
        "CrossNER_music",
        "CrossNER_politics",
        "CrossNER_science",
    ]

    all_results = pd.DataFrame()  # without crossNER

    for eval_ds in tqdm(all_paths):
        if not (
            only_zero_shot
            and any([zs_benchmark in eval_ds for zs_benchmark in zero_shot_benc])
        ):
            continue

        dataset_name = eval_ds.split("/")[-1]

        if "sample_" not in eval_ds:
            metrics = get_for_one_path(eval_ds, model, dataloader, synonyms)

            if metrics is not None:
                metrics["zeroshot"] = (
                    True
                    if any(
                        [
                            zs_benchmark in dataset_name
                            for zs_benchmark in zero_shot_benc
                        ]
                    )
                    else False
                )

                # log the results
                save_path = os.path.join(log_dir, f"{dataset_name}.txt")

                # write to file
                with open(save_path, "a") as f:
                    f.write(metrics.to_string() + "\n")

                all_results = pd.concat([all_results, metrics])

    all_results.reset_index(drop=True, inplace=True)

    if synonyms:
        save_metrics_synonyms(all_results, log_dir)
    else:
        save_metrics(all_results, log_dir)


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
