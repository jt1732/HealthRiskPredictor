import pandas as pd
from sklearn.model_selection import train_test_split

DATA_ENTRIES = 70000

cardio_csv = pd.read_csv('modelv2/cardio.csv')
assert len(cardio_csv) == DATA_ENTRIES, f"The cardio.csv file should contain {DATA_ENTRIES} rows."

train_df, test_df = train_test_split(cardio_csv, test_size=0.2, random_state=42)


cardio_info = cardio_csv.info()
print(cardio_info)

"""
Cardio csv has loaded with correct number of entries and columns.
Cardio info tells us that all rows have non-null values and the data types of each column are consistent
"""