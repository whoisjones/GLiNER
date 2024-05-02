import json


def load_litset():
    with open("/vol/tmp/goldejon/gliner/train_datasets/litset.jsonl", "r") as f:
        data = []
        for line in f.readlines():
            data.append(json.loads(line))
        data = [x for x in data if x["ner"]]

    return data
