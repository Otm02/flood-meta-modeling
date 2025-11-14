import polars as pl
import numpy as np
import pathlib
from rich.console import Console
from rich.table import Table
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
)

console = Console()
use_mean = True
ft_mean_dir = pathlib.Path("data/finetuning_results_mean.csv")
ft_all_dir = pathlib.Path("data/finetuning_results_mean.csv")
ft = pl.read_csv(ft_mean_dir if use_mean else ft_all_dir)
console.print(f"Columns: {len(ft.columns)}")

np.random.seed(42)  
unique_countries = np.sort(ft["country"].unique())
kf = KFold(n_splits=3, shuffle=True)

fold_counter = 0
feature_column_names = ["area", "aridity", "pre_mm_syr", "dis_m3_pyr"]
# Iterate through the folds
for train_index, test_index in kf.split(unique_countries):
    train_countries, test_countries = unique_countries[train_index], unique_countries[test_index]

    console.print(f"Fold number {fold_counter}")
    console.print(f"{train_countries=}")
    console.print(f"{test_countries=}")

    
    train_df = ft.filter(pl.col("country").is_in(train_countries))
    train_features = train_df[feature_column_names].to_numpy()
    train_target = train_df["test_delta"].to_numpy()

    test_df = ft.filter(pl.col("country").is_in(test_countries))
    test_features = test_df[feature_column_names].to_numpy()
    test_target = test_df["test_delta"].to_numpy()
    console.print(f"{train_features.shape=}")
    console.print(f"{test_features.shape=}")

    console.print(f"train target mean: {train_target.mean()}")
    console.print(f"test target mean: {test_target.mean()}")

    reg = linear_model.LinearRegression().fit(train_features, train_target)
    # reg = linear_model.Ridge(alpha=0.4, random_state=42).fit(train_features, train_target)
    train_model_preds = reg.predict(train_features)
    test_model_preds = reg.predict(test_features)

    results_table = Table(title="Results")
    results_table.add_column("Metric")
    results_table.add_column("Train")
    results_table.add_column("Test")
    results_table.add_row(
        "MSE",
        f"{mean_squared_error(train_model_preds, train_target):.4f}",
        f"{mean_squared_error(test_model_preds, test_target):.4f}",
    )
    results_table.add_row(
        "RMSE",
        f"{root_mean_squared_error(train_model_preds, train_target):.4f}",
        f"{root_mean_squared_error(test_model_preds, test_target):.4f}",
    )

    results_table.add_row(
        "MAE",
        f"{mean_absolute_error(train_model_preds, train_target):.4f}",
        f"{mean_absolute_error(test_model_preds, test_target):.4f}",
    )

    results_table.add_row(
        "R2",
        f"{r2_score(train_model_preds, train_target):.4f}",
        f"{r2_score(test_model_preds, test_target):.4f}",
    )
    console.print(results_table)

    fold_counter += 1
    console.print(f"-"*80)

console.print(f"Train/Test Split on latitude")
train_df = ft.filter(pl.col("gauge_lat") > 0)
train_features = train_df[feature_column_names].to_numpy()
train_target = train_df["test_delta"].to_numpy()

test_df = ft.filter(pl.col("gauge_lat") < 0)
test_features = test_df[feature_column_names].to_numpy()
test_target = test_df["test_delta"].to_numpy()
console.print(f"{train_features.shape=}")
console.print(f"{test_features.shape=}")

console.print(f"train target mean: {train_target.mean()}")
console.print(f"test target mean: {test_target.mean()}")

reg = linear_model.LinearRegression().fit(train_features, train_target)
# reg = linear_model.Ridge(alpha=0.4, random_state=42).fit(train_features, train_target)
train_model_preds = reg.predict(train_features)
test_model_preds = reg.predict(test_features)

results_table = Table(title="Results")
results_table.add_column("Metric")
results_table.add_column("Train")
results_table.add_column("Test")
results_table.add_row(
    "MSE",
    f"{mean_squared_error(train_model_preds, train_target):.4f}",
    f"{mean_squared_error(test_model_preds, test_target):.4f}",
)
results_table.add_row(
    "RMSE",
    f"{root_mean_squared_error(train_model_preds, train_target):.4f}",
    f"{root_mean_squared_error(test_model_preds, test_target):.4f}",
)

results_table.add_row(
    "MAE",
    f"{mean_absolute_error(train_model_preds, train_target):.4f}",
    f"{mean_absolute_error(test_model_preds, test_target):.4f}",
)

results_table.add_row(
    "R2",
    f"{r2_score(train_model_preds, train_target):.4f}",
    f"{r2_score(test_model_preds, test_target):.4f}",
)
console.print(results_table)
