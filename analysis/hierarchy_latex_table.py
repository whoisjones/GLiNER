import pandas as pd
import sys

sys.path.append("/vol/fob-vol7/mi18/goldejon/GLiNER")

from evaluate_hierarchy import hierarchy


def print_table():
    df = pd.DataFrame.from_dict(hierarchy)
    df = df.transpose().reset_index()
    df.columns = [
        "Top-Level",
        "Subclass",
        "Subclass",
        "Subclass",
        "Subclass",
        "Subclass",
    ]
    print(df.to_latex(column_format="l|lllll", index=False))


if __name__ == "__main__":
    print_table()
