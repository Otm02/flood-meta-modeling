# %%
import polars as pl
import pathlib
from rich.console import Console
from rich.table import Table

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
table1 = Table(title="Table 1 from Paper")
table1.add_column("Model")
table1.add_column("NSE (mean)")
table1.add_column("NSE (median)")
table1.add_row(
    "Pre-Train",
    f"{ft_all['test_origin'].mean():.3f}",
    f"{ft_all['test_origin'].median():.3f}",
)
table1.add_row(
    "Finetuned",
    f"{(ft_all['test_origin'] + ft_all['test_delta']).mean():.3f}",
    f"{(ft_all['test_origin'] + ft_all['test_delta']).median():.3f}",
)
# where are the KGE (mean) and KGE (median)?
console.print(table1)
# %%
