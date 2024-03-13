import json
from typing import Union, Dict
from tqdm import tqdm

import datasets
from datasets import load_dataset


def verbalize_labels(
    dataset: datasets.Dataset,
    dataset_name: str,
    label_column: str,
) -> datasets.Dataset:
    """Verbalizes the labels in the dataset."""
    type_verbalizations = load_dataset(
        "json", data_files="data/type_descriptions.json", split="train"
    )
    # Filter for the dataset and label column
    type_verbalizations = type_verbalizations.filter(
        lambda example: (example["dataset_name"] == dataset_name)
        and (example["label_column"] == label_column)
    )
    # Extract current labels
    labels = dataset.features[label_column].feature.names

    # If current labels are in BIO format we need to add the verbalizations in BIO format as well
    if any([label.startswith("B-") or label.startswith("I-") for label in labels]):

        def add_bio_tags(examples):
            bio_type_verbalizations = {}
            for key, value in examples.items():
                if key in ["original_label", "verbalized_label"]:
                    bio_type_verbalizations[key] = [
                        "B-" + label for label in examples[key]
                    ] + ["I-" + label for label in examples[key]]
                else:
                    bio_type_verbalizations[key] = value * 2
            return bio_type_verbalizations

        type_verbalizations = type_verbalizations.map(add_bio_tags, batched=True)

    for type_verbalization in type_verbalizations:
        idx = labels.index(type_verbalization["original_label"])
        labels[idx] = type_verbalization["verbalized_label"]

    dataset = dataset.cast_column(
        label_column,
        datasets.Sequence(datasets.ClassLabel(names=labels)),
    )

    return dataset


def format_dataset(dataset: datasets.Dataset, label_column: str) -> datasets.Dataset:
    """Formats the OntoNotes dataset to a list of dictionaries with tokenized text and named entity spans."""
    id2label = dict(enumerate(dataset.features[label_column].feature.names))
    formatted_dataset = []
    for i, example in enumerate(tqdm(dataset)):
        data_point = {"tokenized_text": example["tokens"]}

        type_id = None
        start = None
        end = None

        current_spans = []
        for word_id, tag in enumerate(example[label_column]):
            # Skip any token that is not part of an entity.
            if tag == 0:
                # If we have a start, a non-entity will end the current entity.
                if start is not None:
                    current_spans.append([start, end, id2label.get(type_id)])
                    start = None
                    end = None
                    type_id = None
                continue
            # We must have found a tag, but we don't have a start yet -> Start a new entity.
            elif start is None:
                type_id = tag
                start = word_id
                end = word_id
            # We must have found a tag, but we have a start. If the type is identical to current tag -> Continue the entity.
            elif tag == type_id:
                end = word_id
            # We must have found a tag, we have a start and the previous type is different to current tag -> End current entity and start a new one.
            else:
                current_spans.append([start, end, id2label.get(type_id)])
                start = word_id
                end = word_id
                type_id = tag

        # At last, if we have a start, we must end the entity.
        if start is not None:
            current_spans.append([start, end, id2label.get(type_id)])

        data_point["ner"] = current_spans
        formatted_dataset.append(data_point)
    return formatted_dataset


if __name__ == "__main__":
    dataset_name = "DFKI-SLT/few-nerd"
    dataset_config = "supervised"
    label_column = "fine_ner_tags"

    fewnerd = load_dataset(dataset_name, dataset_config, split="train")
    labels = fewnerd.features[label_column].feature.names

    fewnerd = verbalize_labels(fewnerd, dataset_name, label_column)
    formatted_fewnerd = format_dataset(fewnerd, label_column)

    with open("fewnerd.json", "w") as f:
        json.dump(formatted_fewnerd, f)
