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
    labels = dataset.features[label_column].feature.names
    new_labels = [
        (
            label.split("-")[-1]
            if label.split("-")[-1] != "other"
            else label.split("-")[0] + " (other)"
        )
        for label in labels
    ]

    dataset = dataset.cast_column(
        label_column,
        datasets.Sequence(datasets.ClassLabel(names=new_labels)),
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

    with open("fewnerd_v2.json", "w") as f:
        json.dump(formatted_fewnerd, f)
