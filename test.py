import glob

import pandas as pd

runs = glob.glob("logs/*")

for run in runs:
    files = glob.glob(run + "/*")
    for file in files:
        if file.endswith(".pkl"):
            df = pd.read_pickle(file)
        else:
            with open(file) as f:
                lines = f.readlines()
            parts = lines[0].split(":")
            model = parts[1].strip()

    print(50 * "-")
    print(model)
    df = df[df['k'] == 1000]
    df = df[df['similarity_type'] == "Weighted Average"]
    output = (
        df[["train_dataset", "similarity"]]
        .groupby(["train_dataset"])
        .mean()
        .reset_index()
    )
    print(output)
    print(50 * "-")
