from data import get_eval_datasets_stats, get_train_datasets_stats

if __name__ == "__main__":
    train_statistics = get_train_datasets_stats(
        base_path="/home/ec2-user/paper_data",
    )

    eval_statistics = get_eval_datasets_stats(base_path="/home/ec2-user/paper_data")
