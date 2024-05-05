import pandas as pd
import sys

sys.path.append("/vol/fob-vol7/mi18/goldejon/GLiNER")

from evaluate_synonyms import synonyms


def print_table():
    df = pd.DataFrame.from_dict(synonyms)
    df = df.transpose()
    df.columns = ["Original", "Synonym", "Synonym", "Synonym", "Synonym"]
    print(df.to_latex(column_format="l|llll", index=False))


if __name__ == "__main__":
    print_table()
