import polars as pl
import pathlib
from rich.console import Console

console = Console()

root = pathlib.Path.cwd()
ft_all = pl.read_csv(root / "data/finetuning_results_all.csv")
ft_mean = pl.read_csv(root / "data/finetuning_results_mean.csv")

basins = ft_mean[["country", "gauge_name", "basin"]]
console.print(f"Number of Finetuned Basins: {basins.shape[0]}", style="bold red")

basins_per_country = basins["country"].value_counts()

console.print(basins_per_country)


console.print("What is 'finetuning_results_all'?")
console.print("-> 8 Random instances of the Model")
console.print("-> 152 River Basins")
console.print(f"=> 8*152={ft_all.shape[0]:,} Datapoints")
# print(ft_all.describe())
# print(ft_all["basin"].value_counts())
# print(ft_mean[["country", "gauge_name", "basin"]])
