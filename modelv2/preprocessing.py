from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

NUMERIC = ["age", "height", "weight", "bmi", "ap_hi", "ap_lo"]
CATEGORICAL = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]


def remove_invalid_rows(df):
    start = len(df)

    before = len(df)
    df = df[~df.drop(columns="id").duplicated()] 
    print("duplicates:", before - len(df))

    before = len(df)
    df = df[df["height"].between(130, 210)]
    print("height outside 130-210:", before - len(df))

    before = len(df)
    df = df[df["weight"] >= 40]
    print("weight under 40:", before - len(df))

    before = len(df)
    bmi = df["weight"] / (df["height"] / 100) ** 2
    df = df[bmi.between(12, 80)]
    print("BMI outside 12-80:", before - len(df))

    before = len(df)
    df = df[df["ap_hi"].between(50, 250)]
    print("ap_hi outside 50-250:", before - len(df))

    before = len(df)
    df = df[df["ap_lo"].between(30, 200)]
    print("ap_lo outside 30-200:", before - len(df))

    before = len(df)
    df = df[df["ap_lo"] < df["ap_hi"]]
    print("ap_lo >= ap_hi:", before - len(df))

    print("total removed:", start - len(df), "of", start)
    return df


def add_features(df):
    df = df.copy()  # work on a copy so the original DataFrame isn't changed
    df["age"] = df["age"] / 365.25
    df["bmi"] = df["weight"] / (df["height"] / 100) ** 2
    return df.drop(columns="id")


def build_preprocessor():
    #Scale the numeric columns
    return ColumnTransformer([
        ("scale", StandardScaler(), NUMERIC),
        ("keep", "passthrough", CATEGORICAL),
    ])


if __name__ == "__main__":
    from data import split_data

    train_df, test_df = split_data()
    print("Train:")
    train_clean = remove_invalid_rows(train_df)
    """
    Applying the rules to the test set isn't leakage: the rules were chosen from train only, and nothing
    is learned from the test data.
    """
    print("\nTest:")
    test_clean = remove_invalid_rows(test_df)

    train_features = add_features(train_clean)
    print("\nColumns after add_features:", list(train_features.columns))
    print(train_features[["age", "bmi"]].describe().round(1))

    test_features = add_features(test_clean)
    preprocessor = build_preprocessor()
    train_scaled = preprocessor.fit_transform(train_features)  # learn mean/std from train, then apply
    test_scaled = preprocessor.transform(test_features)  # apply train's mean/std to test, learn nothing
    print("\nMean of each column after scaling (numeric should be about 0):")
    print("train:", train_scaled.mean(axis=0).round(2))
    print("test: ", test_scaled.mean(axis=0).round(2))
