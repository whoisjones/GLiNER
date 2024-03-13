import json
from typing import Dict
from tqdm import tqdm

import datasets
from datasets import load_dataset


def format_dataset(
    dataset: datasets.Dataset, id2label: Dict, label_column: str
) -> datasets.Dataset:
    """Formats the dataset to a list of dictionaries with tokenized text and named entity spans."""
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
    dataset_name = "whoisjones/litset"
    label_column = "ner_tags"

    litset = load_dataset(dataset_name, split="train")

    with open("/vol/tmp/goldejon/ner4all/loner/labelID2label.json", "r") as f:
        id2label = json.load(f)
    id2label = {int(k): v for k, v in id2label.items()}

    new_id2label = {}
    for k, v in id2label.items():
        annotation = {}
        if "labels" in v:
            annotation["labels"] = v["labels"]
        else:
            annotation["labels"] = []
        if "description" in v:
            annotation["description"] = [v["description"]]
        else:
            annotation["description"] = []
        new_id2label[k] = annotation

    formatted_litset = format_dataset(litset, new_id2label, label_column)

    with open("litset.json", "w") as f:
        json.dump(formatted_litset, f)
