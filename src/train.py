# %%
import polars as pl
import numpy as np
import pathlib
from rich.console import Console
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

console = Console()
ft_mean_dir = pathlib.Path("data/finetuning_results_mean.csv")
ft_all_dir = pathlib.Path("data/finetuning_results_mean.csv")
ft = pl.read_csv(ft_all_dir)
console.print(f"Columns: {len(ft.columns)}")
# %%
np.random.seed(42)
unique_countries = ft["country"].unique().to_numpy()
np.random.shuffle(unique_countries)

# split countries: 70/30
split_idx = int(len(unique_countries) * 0.7)
train_countries = unique_countries[:split_idx]
test_countries = unique_countries[split_idx:]
console.print(f"{train_countries=}")
console.print(f"{test_countries=}")

feature_column_names = ["aridity"]  # ["area"]  # ['area', "aridity"]
train_df = ft.filter(pl.col("country").is_in(train_countries))
train_features = train_df[feature_column_names].to_numpy()
train_target = train_df["test_delta"].to_numpy()

test_df = ft.filter(pl.col("country").is_in(test_countries))
test_features = test_df[feature_column_names].to_numpy()
test_target = test_df["test_delta"].to_numpy()
console.print(f"{train_features.shape=}")
console.print(f"{test_features.shape=}")


# %%
reg = linear_model.LinearRegression().fit(train_features, train_target)
model_preds = reg.predict(test_features)

console.print(f"Mean squared error: {mean_squared_error(model_preds, test_target):.3f}")
console.print(
    f"Mean absolute error: {mean_absolute_error(model_preds, test_target):.3f}"
)
console.print(f"Coefficient of determination: {r2_score(model_preds, test_target):.3f}")
