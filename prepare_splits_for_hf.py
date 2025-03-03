from compute_new_splits import create_splits
import json
import pandas as pd
from datasets import Dataset, DatasetDict


def convert_to_hf_format(data_point):
    tags = ["O"] * len(data_point["tokenized_text"])
    spans = []
    for ent in data_point["ner"]:
        start, end, label = ent[0], ent[1], ent[2]
        spans.append({"start": start, "end": end, "label": label})
        if start == end:
            tags[start] = "B-" + label
        else:
            try:
                tags[start] = "B-" + label
                tags[start + 1 : end + 1] = ["I-" + label] * (end - start)
            except:
                pass
    return {"tokens": data_point["tokenized_text"], "ner_tags": tags, "spans": spans}


if __name__ == "__main__":
    for dataset in ["pilener_train"]:
        train_data_dir = f"/vol/tmp/goldejon/gliner/train_datasets/{dataset}.json"
        with open(train_data_dir, "r") as f:
            data = json.load(f)
        for filter_by in ["entropy", "max"]:
            dataset_dict = DatasetDict()
            for setting in ["easy", "medium", "hard"]:
                new_split = create_splits(
                    data,
                    dataset,
                    filter_by=filter_by,
                    setting=setting,
                )

                hf_format = [
                    convert_to_hf_format(data_point) for data_point in new_split
                ]

                ds = Dataset.from_pandas(pd.DataFrame(data=hf_format))
                dataset_dict[setting] = ds

            dataset_dict.push_to_hub(
                f"{dataset.replace('_train', '')}_{filter_by}_splits"
            )
