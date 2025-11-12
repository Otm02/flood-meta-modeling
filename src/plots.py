# %%
import polars as pl
import matplotlib.pyplot as plt
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
# %%

plt.scatter(ft_all["area"], ft_all["test_delta"])
plt.xlabel("area")
plt.xlabel("test_delta")
plt.show()
# %%

# plt.boxplot(ft_mean["test_origin"])
plt.boxplot(ft_all["test_delta"])
# plt.scatter(ft_mean["test_origin"], ft_mean["test_delta"])
plt.show()
