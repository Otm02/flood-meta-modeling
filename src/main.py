import polars as pl
import pathlib

root = pathlib.Path.cwd()
ft_all = pl.read_csv(root / "data/finetuning_results_all.csv")
ft_mean = pl.read_csv(root / "data/finetuning_results_mean.csv")

print(ft_all.describe())
print(ft_all["basin"].value_counts())
print(ft_mean[["country", "gauge_name", "basin"]])
