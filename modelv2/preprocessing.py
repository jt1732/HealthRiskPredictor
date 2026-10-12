def remove_invalid_rows(df):
    """Drop duplicate and implausible rows"""
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
