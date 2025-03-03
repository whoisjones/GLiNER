import glob

import pandas as pd

from analysis.data import display_eval, display_train


def get_mean_std(base_path: str = "/vol/tmp/goldejon/gliner"):
    paths = glob.glob(f"{base_path}/eval/*/*/results.pkl")

    all_results = pd.DataFrame()
    for path in paths:
        result = pd.read_pickle(path)
        metadata = path.split("/")
        train_dataset = metadata[-3]
        seed = metadata[-2]
        result["train_dataset"] = train_dataset
        result["seed"] = seed
        all_results = pd.concat([all_results, result])

    all_results = all_results[all_results["entity"] == "average"]
    all_results = all_results.reset_index(drop=True)

    all_results["train_dataset"] = all_results["train_dataset"].apply(
        lambda x: display_train[x]
    )

    all_results["eval_benchmark"] = all_results["eval_benchmark"].apply(
        lambda x: display_eval[x]
    )

    return all_results


if __name__ == "__main__":
    mean_std = get_mean_std("/vol/tmp/goldejon/gliner")
    print()
