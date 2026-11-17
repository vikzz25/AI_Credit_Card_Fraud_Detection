import pandas as pd

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT_DIR / "data" / "raw" / "creditcard.csv"


def load_dataset():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    return df


if __name__ == "__main__":

    df = load_dataset()

    print("\nDataset Loaded Successfully")

    print("\nFirst Five Records:")

    print(df.head())

    print("\nDataset Shape:")

    print(df.shape)

    print("\nDataset Information:")

    print(df.info())

    print("\nClass Distribution:")

    print(df["Class"].value_counts())