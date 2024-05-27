import os
import json
import random
import pandas as pd
from collections import Counter
from data import display_train, get_eval_datasets_stats


def get_train_labels(
    train_dataset: str,
):
    train_datasets_path: str = "/vol/tmp/goldejon/gliner/train_datasets"
    if train_dataset == "litset":
        with open("/vol/tmp/goldejon/ner4all/loner/labelID2label.json", "r") as f:
            id2label = json.load(f)
        id2label.pop("0")

        data = [label for label in id2label.values()]

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

        labels = []
        for dp in data:
            for entity in dp["ner"]:
                labels.append(random.choice(entity[-1]))

    else:
        with open(os.path.join(train_datasets_path, f"{train_dataset}.json"), "r") as f:
            data = json.load(f)

        labels = []
        for dp in data:
            for entity in dp["ner"]:
                labels.append(entity[-1])

    return labels


def get_train_datasets_stats():
    if os.path.exists("/vol/tmp/goldejon/gliner/analysis/full_train_statistics.pkl"):
        return pd.read_pickle(
            "/vol/tmp/goldejon/gliner/analysis/full_train_statistics.pkl"
        )

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
        if train_dataset == "litset":
            with open("/vol/tmp/goldejon/gliner/train_datasets/litset.jsonl", "r") as f:
                data = []
                for line in f.readlines():
                    data.append(json.loads(line))
                dataset = [x for x in data if x["ner"]]
        else:
            with open(
                f"/vol/tmp/goldejon/gliner/train_datasets/{train_dataset}.json", "r"
            ) as f:
                dataset = json.load(f)
        len_train_dataset = len(dataset)
        train_labels = get_train_labels(train_dataset)
        train_labels = [label.lower() for label in train_labels]

        train_labels_binary = set(train_labels)
        train_labels_count = Counter(train_labels)

        df = pd.DataFrame(
            {
                "train_dataset": [train_dataset],
                "num_sentences": [len_train_dataset],
                "train_labels_set": [train_labels_binary],
                "train_labels_counter": [train_labels_count],
            }
        )

        train_datasets_stats = pd.concat([train_datasets_stats, df])

    train_datasets_stats.reset_index(drop=True, inplace=True)

    train_datasets_stats.to_pickle(
        "/vol/tmp/goldejon/gliner/analysis/full_train_statistics.pkl"
    )

    return train_datasets_stats


if __name__ == "__main__":
    train_statistics = get_train_datasets_stats()
    train_statistics["entities_per_sentence"] = train_statistics.apply(
        lambda row: f"{(sum(row['train_labels_counter'].values()) / row['num_sentences']):.1f}",
        axis=1,
    )
    train_statistics["train_labels_set"] = train_statistics["train_labels_set"].apply(
        lambda x: f"{len(x)}" if len(x) < 100 else f"{len(x) / 1000:.1f}k"
    )
    train_statistics["num_sentences"] = train_statistics["num_sentences"].apply(
        lambda x: f"{x / 1000:.1f}k"
    )
    train_statistics["train_dataset"] = train_statistics["train_dataset"].apply(
        lambda x: display_train[x]
    )
    train_statistics.drop(columns=["train_labels_counter"], inplace=True)

    train_statistics.columns = [
        "FT-Dataset",
        "# Sentences",
        "# Entities",
        "$\diameter$ Ent. per Sent.",
    ]
    print(train_statistics.to_latex(index=False))

    eval_statistics = get_eval_datasets_stats()
    print(f"Total Eval Labels: {len(list(set().union(*eval_statistics["eval_labels_set"].tolist())))}")
