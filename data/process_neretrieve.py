import json
from tqdm import tqdm
from datasets import Dataset


def main():
    paths = [
        "/vol/tmp/goldejon/neretrieve/NERetrive_sup_train.jsonl",
        "/vol/tmp/goldejon/neretrieve/NERetrive_sup_test.jsonl",
    ]

    for path in paths:
        data = []
        with open(path, "r") as f:
            lines = f.readlines()
            with tqdm(total=len(lines), position=0, leave=True) as pbar:
                for line in tqdm(lines):
                    pbar.update()
                    data_point = json.loads(line)

                    annotations = []
                    for label, mentions in data_point["tagged_entities"].items():
                        for _, spans in mentions.items():
                            for _, span_list in spans.items():
                                for span in span_list:
                                    if not span:
                                        continue
                                    annotations.append(
                                        (
                                            span[0],
                                            span[1] if len(span) == 2 else span[0],
                                            label.lower().replace("_", " "),
                                        )
                                    )
                    annotations = sorted(annotations, key=lambda x: x[0])

                    pointer = None
                    merged_annotations = []
                    for start, end, label in annotations:
                        if pointer is None:
                            pointer = [start, end, [label]]
                        elif start <= pointer[1]:
                            extended_label = pointer[-1]
                            extended_label.append(label)
                            pointer = [start, end, extended_label]
                        else:
                            merged_annotations.append(pointer)
                            pointer = [start, end, [label]]

                    data.append(
                        {
                            "tokenized_text": data_point["document_token_sequence"],
                            "ner": merged_annotations,
                        }
                    )

        if "train" in path:
            extension = "train"
        else:
            extension = "test"

        with open(
            f"/vol/tmp/goldejon/gliner/train_datasets/neretrieve_{extension}.json", "w"
        ) as f:
            json.dump(data, f)


if __name__ == "__main__":
    main()
