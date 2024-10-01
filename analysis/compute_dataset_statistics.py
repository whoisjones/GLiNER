from data import get_eval_datasets_stats, get_train_datasets_stats

if __name__ == "__main__":
    train_statistics = get_train_datasets_stats(
        base_path="/vol/tmp/goldejon/gliner",
    )
    train_statistics["total_entities"] = train_statistics[
        "train_labels_set_full_dataset"
    ].apply(lambda x: len(x))
    train_statistics["total_mentions"] = train_statistics[
        "train_labels_counter_full_dataset"
    ].apply(lambda x: sum(x.values()))
    train_statistics["avg_entities_per_sentence"] = (
        train_statistics["total_mentions"] / train_statistics["num_sentences"]
    )
    train_statistics = train_statistics.drop(
        columns=[
            "train_labels_counter_sampled",
            "train_labels_set_sampled",
            "train_labels_counter_full_dataset",
            "train_labels_set_full_dataset",
        ]
    )
    print(train_statistics)

    eval_statistics = get_eval_datasets_stats(base_path="/vol/tmp/goldejon/gliner")
    eval_statistics["total_entities"] = eval_statistics["eval_labels_set"].apply(
        lambda x: len(x)
    )
    eval_statistics["total_mentions"] = eval_statistics["eval_labels_counter"].apply(
        lambda x: sum(x.values())
    )
    eval_statistics["avg_entities_per_sentence"] = (
        eval_statistics["total_mentions"] / eval_statistics["num_sentences"]
    )
    eval_statistics = eval_statistics.drop(
        columns=["eval_labels_counter", "eval_labels_set"]
    )
    print(eval_statistics)
