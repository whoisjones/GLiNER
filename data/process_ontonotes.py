import json
from tqdm import tqdm

import datasets
from datasets import load_dataset


def flatten(examples):
    sentences = [sentence for doc in examples["sentences"] for sentence in doc]
    examples["tokens"] = [sentence["words"] for sentence in sentences]
    examples["ner_tags"] = [sentence["named_entities"] for sentence in sentences]
    return examples


def to_io_format(ontonotes, labels):

    id2biolabel = {idx: label for idx, label in enumerate(labels)}
    id2iolabel = {}
    label_mapping = {}
    for idx, label in id2biolabel.items():
        if label == "O":
            id2iolabel[len(id2iolabel)] = label
            label_mapping[idx] = len(id2iolabel) - 1
        elif label.startswith("B-") or label.startswith("I-"):
            io_label = label[2:]
            if io_label not in id2iolabel.values():
                io_label_id = len(id2iolabel)
                id2iolabel[io_label_id] = io_label
                label_mapping[idx] = io_label_id
            else:
                label_mapping[idx] = [
                    k for k, v in id2iolabel.items() if v == io_label
                ][0]

    def io_format(examples):
        examples["ner_tags"] = [
            [label_mapping.get(old_id) for old_id in sample]
            for sample in examples["ner_tags"]
        ]
        return examples

    ontonotes = ontonotes.map(io_format, batched=True, desc="Convert to IO format.")
    ontonotes = ontonotes.cast_column(
        "ner_tags",
        datasets.Sequence(datasets.ClassLabel(names=list(id2iolabel.values()))),
    )

    return ontonotes


def verbalize_labels(ontonotes, dataset_name, label_column):
    type_verbalizations = load_dataset(
        "json", data_files="data/type_descriptions.json", split="train"
    )
    # Filter for the dataset and label column
    type_verbalizations = type_verbalizations.filter(
        lambda example: (example["dataset_name"] == dataset_name)
        and (example["label_column"] == label_column)
    )
    # Extract current labels
    labels = ontonotes.features[label_column].feature.names

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

    ontonotes = ontonotes.cast_column(
        "ner_tags",
        datasets.Sequence(datasets.ClassLabel(names=labels)),
    )

    return ontonotes


def format_ontonotes(ontonotes):
    id2label = dict(enumerate(ontonotes.features["ner_tags"].feature.names))
    formatted_ontonotes = []
    for i, example in enumerate(tqdm(ontonotes)):
        data_point = {"tokenized_text": example["tokens"]}

        type_id = None
        start = None
        end = None

        current_spans = []
        for word_id, tag in enumerate(example["ner_tags"]):
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
        formatted_ontonotes.append(data_point)
    return formatted_ontonotes


if __name__ == "__main__":
    dataset_name = "conll2012_ontonotesv5"
    dataset_config = "english_v4"
    label_column = "ner_tags"

    ontonotes = load_dataset(dataset_name, dataset_config, split="train")
    labels = ontonotes.features["sentences"][0]["named_entities"].feature.names

    ontonotes = ontonotes.map(
        flatten,
        batched=True,
        remove_columns=ontonotes.column_names,
        desc="Flatten OntoNotes.",
    )

    ontonotes = to_io_format(ontonotes, labels)
    ontonotes = verbalize_labels(ontonotes, dataset_name, label_column)
    formatted_ontonotes = format_ontonotes(ontonotes)

    with open("ontonotes.json", "w") as f:
        json.dump(formatted_ontonotes, f)
