from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

DATA_PATH = Path(__file__).parent / "cardio.csv"
DATA_ENTRIES = 70000
TARGET = "cardio"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data(path=DATA_PATH):
    cardio_csv = pd.read_csv(path)
    assert len(cardio_csv) == DATA_ENTRIES, f"The cardio.csv file should contain {DATA_ENTRIES} rows."
    return cardio_csv


def split_data(cardio_csv=None):
    if cardio_csv is None:
        cardio_csv = load_data()

    train_idx, test_idx = next(
        StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        .split(cardio_csv, cardio_csv[TARGET])
    )

    train_df = cardio_csv.iloc[train_idx]
    test_df = cardio_csv.iloc[test_idx]
    return train_df, test_df


if __name__ == "__main__":
    cardio_csv = load_data()
    cardio_csv.info()

    train_df, test_df = split_data(cardio_csv)
    print(f"\nTrain: {len(train_df)} rows, cardio rate {train_df[TARGET].mean():.4f}")
    print(f"Test:  {len(test_df)} rows, cardio rate {test_df[TARGET].mean():.4f}")

