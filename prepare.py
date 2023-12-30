from math import nan
from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV


def read_set(is_train=True):
    return pd.read_csv(f"datasets/{'train' if is_train else 'test'}.csv")


def expand_cab(x):
    x = str(x)

    if x == "nan":
        return nan

    cabins = x.split(" ")

    deck = sorted([cabin[0] for cabin in cabins])[0]

    rooms = sorted([cabin[1:] for cabin in cabins])[0]

    num_cabins = len(cabins)

    return {"deck": deck, "rooms": rooms, "num_cabins": num_cabins}


def apply_expand_cab(df: pd.DataFrame):
    df[
        ["cabin_highest_class_deck", "cabin_lowest_room_num", "cabin_num_cabins"]
    ] = pd.json_normalize(df["Cabin"].map(expand_cab))


def drop_nan_embarked(df: pd.DataFrame):
    return df[~df["Embarked"].isna()].reset_index()


def encode_embarked(df: pd.DataFrame):
    df["Embarked_S"] = df["Embarked"].map(lambda x: 1 if x == "S" else 0)
    df["Embarked_C"] = df["Embarked"].map(lambda x: 1 if x == "C" else 0)
    df["Embarked_Q"] = df["Embarked"].map(lambda x: 1 if x == "Q" else 0)


def encode_sex(df: pd.DataFrame):
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0})


def get_family_size(df: pd.DataFrame):
    return df["SibSp"] + df["Parch"] + 1


class LogNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X):
        return self

    def _log_normalize(self, x):
        return np.sign(x) * np.log(np.abs(x) + 1)

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(self._log_normalize(np.array(X)), columns=X.columns)
        else:
            return np.where(X >= 0, np.log(X + 1), -np.log(-X + 1))


def scale_col(df: pd.DataFrame, col: str, apply=False, use_scaler=None):
    if use_scaler is None:
        scaler = LogNormalizer()
    else:
        scaler = use_scaler

    ret = scaler.fit_transform(df[[col]])

    if apply:
        df[[col]] = ret

    return scaler


def clean_cabin_lowest_room_num(df: pd.DataFrame):
    df["cabin_lowest_room_num"] = (
        df["cabin_lowest_room_num"].map(lambda x: x if x != "" else 0).astype(float)
    )


def get_title(name: str):
    return name.split(",")[1].split(".")[0].strip()


good_titles = {
    "title_mr": {"Mr", "Master"},
    "title_ms": {"Miss", "Ms", "Mlle"},
    "title_mrs": {"Mrs", "Mme"},
    "title_imp": {"Dr", "Col", "Capt", "Sir", "Lady", "Don"},
    "title_other": {"Rev", "Major", "Jonkheer", "the Countess"},
}


def in_title_category(x):
    for cat, titles in good_titles.items():
        if x in titles:
            return cat


def merge_title_col(df: pd.DataFrame):
    name_title_col = df["Name"].map(get_title).map(in_title_category)
    return pd.concat([df, pd.get_dummies(name_title_col).astype(int)], axis=1)


def prepare_ages_df(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    ages_df = pd.concat([train_df, test_df], axis=0).copy()

    to_drop_cols = [
        "Embarked",
        "Name",
        "Ticket",
        *[c for c in ages_df if "cabin" in c.lower()],
    ]

    ages_df = ages_df.drop(columns=to_drop_cols).reset_index(drop=True)

    return ages_df


def split_ages_df_to_train_and_nan(
    ages_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ages_train = ages_df[ages_df["Age"].notnull()]
    ages_test = ages_df[ages_df["Age"].isnull()]

    ages_train_X = ages_train.drop(columns=["Age"])
    ages_train_y = ages_train["Age"]

    ages_test_X = ages_test.drop(columns=["Age"])

    return ages_train_X, ages_train_y, ages_test_X


def predict_ages(train_df: pd.DataFrame, test_df: pd.DataFrame):
    ages_df = prepare_ages_df(train_df=train_df, test_df=test_df)

    ages_train_X, ages_train_y, ages_test_X = split_ages_df_to_train_and_nan(ages_df)

    X_train, y_train = (
        ages_train_X.drop(columns=["PassengerId", "Survived"]),
        ages_train_y,
    )

    X_test = ages_test_X.drop(columns=["PassengerId", "Survived"])

    estimator = GradientBoostingRegressor(
        n_estimators=200,
        validation_fraction=0.1,
    )

    grid = {
        "max_depth": [5, 10],
        "min_samples_split": [20],
        "min_samples_leaf": [10, 20],
        "max_features": ["log2", "sqrt"],
        "n_iter_no_change": [100],
        "tol": [0.00001],
    }

    grid_search = GridSearchCV(
        estimator,
        grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
        verbose=4,
    )

    grid_search.fit(X_train, y_train)

    best_estimator = grid_search.best_estimator_

    best_estimator.fit(X_train, y_train)

    print(f"Best estimator: {best_estimator}")
    print(f"Best score: {grid_search.best_score_}")

    age_estimator = best_estimator

    ages_test_X["Age"] = age_estimator.predict(X_test)

    to_drop = [
        "Age",
        "PassengerId",
        "cabin_highest_class_deck",
        "cabin_lowest_room_num",
        "cabin_num_cabins",
        "Cabin",
        "Embarked",
        "Name",
        "Ticket",
    ]

    train_df["Age_pred"] = age_estimator.predict(
        train_df.drop(columns=[*to_drop, "Survived"])
    )
    train_df["Age"] = train_df.apply(
        lambda x: x["Age_pred"] if np.isnan(x["Age"]) else x["Age"], axis=1
    )

    test_df["Age_pred"] = age_estimator.predict(test_df.drop(columns=to_drop))
    test_df["Age"] = test_df.apply(
        lambda x: x["Age_pred"] if np.isnan(x["Age"]) else x["Age"], axis=1
    )

    train_df.drop(columns=["Age_pred"], inplace=True)
    test_df.drop(columns=["Age_pred"], inplace=True)

    return train_df, test_df


def get_ready_train_set(df: pd.DataFrame):
    train = df.drop(
        columns=[
            "Name",
            "Ticket",
            "Embarked",
            "PassengerId",
            *[c for c in df if "cabin" in c.lower()],
        ]
    )

    return train


def main():
    train_df = read_set(is_train=True)
    test_df = read_set(is_train=False)

    train_df.fillna({"Cabin": "O0"}, inplace=True)
    test_df.fillna({"Cabin": "O0"}, inplace=True)

    apply_expand_cab(train_df)
    apply_expand_cab(test_df)

    train_df = drop_nan_embarked(train_df)
    test_df = drop_nan_embarked(test_df)

    encode_embarked(train_df)
    encode_embarked(test_df)

    encode_sex(train_df)
    encode_sex(test_df)

    train_df["Family_Size"] = get_family_size(train_df)
    test_df["Family_Size"] = get_family_size(test_df)

    fare_scaler = scale_col(train_df, "Fare", apply=True)
    scale_col(test_df, "Fare", apply=True, use_scaler=fare_scaler)

    fam_size_scaler = scale_col(train_df, "Family_Size", apply=True)
    scale_col(test_df, "Family_Size", apply=True, use_scaler=fam_size_scaler)

    clean_cabin_lowest_room_num(train_df)
    clean_cabin_lowest_room_num(test_df)

    train_df = merge_title_col(train_df)
    test_df = merge_title_col(test_df)

    train_df.drop(columns=["index"], inplace=True)
    test_df.drop(columns=["index"], inplace=True)

    test_df["Fare"] = test_df["Fare"].interpolate(method="from_derivatives")

    train_df, test_df = predict_ages(train_df, test_df)

    age_scaler = scale_col(train_df, "Age", apply=True)
    scale_col(test_df, "Age", apply=True, use_scaler=age_scaler)

    deck_map = pd.DataFrame(train_df["cabin_highest_class_deck"].value_counts().index)
    deck_map["index"] = deck_map.index

    n_to_deck = deck_map.to_dict()["cabin_highest_class_deck"]
    deck_to_n = {v: k for k, v in n_to_deck.items()}

    train_df["cabin_highest_class_deck"] = train_df["cabin_highest_class_deck"].map(
        deck_to_n
    )
    test_df["cabin_highest_class_deck"] = test_df["cabin_highest_class_deck"].map(
        deck_to_n
    )

    train_df.to_csv("datasets/train_clean.csv", index=False)
    test_df.to_csv("datasets/test_clean.csv", index=False)


if __name__ == "__main__":
    main()
