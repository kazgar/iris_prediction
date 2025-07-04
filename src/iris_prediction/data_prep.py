import polars as pl
from sklearn.model_selection import train_test_split
from pathlib import Path
import os

def split_data(
    X: pl.DataFrame, y: pl.Series, train_size: pl.Float64, val_set: bool
) -> tuple:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, stratify=y
    )
    if not val_set:
        return (X_train, X_test, y_train, y_test)
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, train_size=0.5, stratify=y_test
    )
    return (X_train, X_val, X_test, y_train, y_val, y_test)


def expand_dataset(data: pl.DataFrame) -> pl.DataFrame:
    return data.with_columns(
        (pl.col("sepal_length") + pl.col("sepal_width")).alias("sepal_sum"),
        (pl.col("petal_length") + pl.col("petal_width")).alias("petal_sum"),
    ).with_columns((pl.col("sepal_sum") + pl.col("petal_sum")).alias("total_sum"))


def split_x_y(data: pl.DataFrame) -> tuple:
    X = data.drop("class")
    y = data.get_column("class").to_frame()
    return (X, y)


def transform_y(y: pl.DataFrame) -> tuple:
    unique_y = y["class"].unique().to_list()
    y_mapping = {species: i for i, species in enumerate(unique_y)}
    y = y.with_columns(
        pl.col("class").replace_strict(y_mapping).alias("class")
    ).to_series()
    return (y, y_mapping)

def concat_results(data:dict) -> pl.DataFrame:
    return (
    pl.concat(
        [data[model] for model in data.keys()], how='vertical'
    )
    )

def save_params(data:dict, filename:Path) -> None:
    params = pl.from_dict(data)
    params.write_csv(filename)