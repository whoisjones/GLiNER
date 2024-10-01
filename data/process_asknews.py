import json


def process_asknews(unprocessed_data_path: str):
    with open(unprocessed_data_path, "r") as f:
        unprocessed_data = json.load(f)

    train_split = unprocessed_data["train"]

    #with open("/home/ec2-user/paper_data/train_datasets/asknews.json", "w") as f:
    #    json.dump(train_split, f)
    
    with open("/vol/tmp/goldejon/gliner/train_datasets/asknews.json", "w") as f:
        json.dump(train_split, f)


if __name__ == "__main__":
    #process_asknews(
    #    unprocessed_data_path="/home/ec2-user/paper_data/train_datasets/asknews_unprocessed.json"
    #)

    process_asknews(
        unprocessed_data_path="/vol/tmp/goldejon/gliner/train_datasets/asknews_unprocessed.json"
    )
