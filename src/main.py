# %%
import polars as pl
import pathlib
from rich.console import Console

console = Console()
ft_all_dir = pathlib.Path("data/finetuning_results_all.csv")
ft_mean_dir = pathlib.Path("data/finetuning_results_mean.csv")
if not ft_all_dir.is_file():
    raise ValueError(f"File {ft_all_dir} not found")

if not ft_mean_dir.is_file():
    raise ValueError(f"File {ft_mean_dir} not found")

ft_all = pl.read_csv(ft_all_dir)
ft_mean = pl.read_csv(ft_mean_dir)

basins = ft_mean[["country", "gauge_name", "basin"]]
console.print(f"Number of Finetuned Basins: {basins.shape[0]}", style="bold red")

basins_per_country = basins["country"].value_counts()

# %%
console.print(basins_per_country)


console.print("What is 'finetuning_results_all'?")
console.print("-> 8 Random instances of the Model")
console.print("-> 152 River Basins")
console.print(f"=> 8*152={ft_all.shape[0]:,} Datapoints")
# print(ft_all.describe())
# print(ft_all["basin"].value_counts())
# print(ft_mean[["country", "gauge_name", "basin"]])

# %%
import matplotlib.pyplot as plt

plt.plot(ft_mean["val_origin"], ft_mean["val_delta"])
plt.show()

# %%
