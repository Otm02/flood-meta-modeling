# %%
import pathlib

import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table
from sklearn import linear_model
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

console = Console()
use_mean = True
ft_mean_dir = pathlib.Path("data/finetuning_results_mean.csv")
ft_all_dir = pathlib.Path("data/finetuning_results_mean.csv")
ft = pl.read_csv(ft_mean_dir if use_mean else ft_all_dir)

# Drop spacial + model related data
drop_cols = [
    "",
    "basin",
    "val_delta",
    "test_origin",
    "val_origin",
    "gauge_name",
    "country",
    "gauge_lat",
    "gauge_lon",
]
ft = ft.drop(drop_cols)

y = ft["test_delta"].to_numpy()
X = ft.drop(["test_delta"])

categorical_cols = [
    "clz_cl_smj",
    "clz_cl_smj",
    "glc_cl_smj",
    "pnv_cl_smj",
    "wet_cl_smj",
    "tbi_cl_smj",
    "tec_cl_smj",
    "fmh_cl_smj",
    "fec_cl_smj",
    "lit_cl_smj",
]
X = X.to_dummies(columns=categorical_cols)

numeric_cols = [
    col
    for col in X.columns
    if not any(col.startswith(prefix + "_") for prefix in categorical_cols)
]
X = X.with_columns(
    [(pl.col(numeric_cols) - pl.col(numeric_cols).mean()) / pl.col(numeric_cols).std()]
)

drop_cols = []
for col in X.columns:
    n_nan = X.select(pl.col(col).is_nan().sum())[0, 0]
    n_inf = X.select(pl.col(col).is_infinite().sum())[0, 0]
    if n_nan > 0 or n_inf > 0:
        print("Dropping: ", col, "NaN:", n_nan, "Inf:", n_inf)
        drop_cols.append(col)

X = X.drop(drop_cols)
feature_names = np.array(X.columns)
X = X.to_numpy()

X_baseline = ft["area", "aridity"].to_numpy()

reg = linear_model.LinearRegression()
sfs = SequentialFeatureSelector(
    reg, n_features_to_select=15, direction="forward", scoring="r2", cv=5
)
sfs.fit(X, y)
X_selected = sfs.transform(X)
selected_names = feature_names[sfs.get_support()]

reg.fit(X_selected, y)
model_preds = reg.predict(X_selected)

reg.fit(X_baseline, y)
baseline_model_preds = reg.predict(X_baseline)

results_table = Table(title="Results")
results_table.add_column("Metric")
results_table.add_column("15 Selected Features")
results_table.add_column("Area and Aridity")
results_table.add_row(
    "MSE",
    f"{mean_squared_error(model_preds, y):.4f}",
    f"{mean_squared_error(baseline_model_preds, y):.4f}",
)
results_table.add_row(
    "RMSE",
    f"{root_mean_squared_error(model_preds, y):.4f}",
    f"{root_mean_squared_error(baseline_model_preds, y):.4f}",
)

results_table.add_row(
    "MAE",
    f"{mean_absolute_error(model_preds, y):.4f}",
    f"{mean_absolute_error(baseline_model_preds, y):.4f}",
)

results_table.add_row(
    "R2",
    f"{r2_score(model_preds, y):.4f}",
    f"{r2_score(baseline_model_preds, y):.4f}",
)
console.print(results_table)
console.print(selected_names)
