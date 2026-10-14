# Optimizing global river flood forecasting via meta-modeling

Fine-tuning a global LSTM flood-forecasting model on local data helps some river basins a lot and
most of them very little, yet it costs compute and expertise every time. This project asks whether
a basin's attributes can predict, before fine-tuning, how much it will gain.

We fit a linear-regression **meta-model** that predicts the fine-tuning gain of each basin from its
Caravan attributes (217 features from HydroATLAS and ERA5-Land), across the 159 basins for which
fine-tuning results are available.

McGill COMP 685 team project (three authors, listed in the report).
**Report:** [PDF](https://athmanebenarous.com/papers/flood-meta-modeling.pdf)

## Results

- **Train and test are split by region** to avoid spatial leakage between nearby basins.
- **Forward feature selection** handles having more features than basins. The global model's own
  performance was the first feature selected on every split.
- **Test R² ranged from 0.76** (South America vs. the rest of the world) **to −1.37** (UK vs. the
  rest of the world), and was negative on 3 of 5 splits.

With this data, basin attributes are weak predictors of when fine-tuning pays off. A larger set
of fine-tuned basins is the main requirement for a useful meta-model.

## Data

`src/data/` holds the per-basin fine-tuning results joined with basin attributes:

- `finetuning_results_mean.csv`: one row per basin (159 rows), averaged over model instances.
- `finetuning_results_all.csv`: one row per basin and model instance (8 instances, 1,272 rows).

Sources: fine-tuning results from E. Ryd and G. Nearing,
[*Fine Flood Forecasts: Incorporating local data into global models through fine-tuning*](https://arxiv.org/abs/2504.12559)
(2025); basin attributes from F. Kratzert et al.,
[*Caravan, a global community dataset for large-sample hydrology*](https://doi.org/10.1038/s41597-023-01975-w)
(Scientific Data, 2023).

## Running it

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cd src                         # scripts read data/ relative to src/
uv run python main.py          # dataset overview: basins per country, pre-train vs fine-tuned NSE
uv run python train.py         # linear meta-model with country-grouped cross-validation
uv run python feature_selection.py   # forward feature selection + linear meta-model
```

The scripts use `# %%` cells, so they also run cell by cell in VS Code or Jupyter.

| Script | What it does |
| --- | --- |
| `src/main.py` | Loads both CSVs and summarizes the data |
| `src/train.py` | Trains and scores the linear meta-model on a small feature set |
| `src/feature_selection.py` | Sequential forward feature selection, then the linear meta-model |
| `src/feature_understanding.py` | Separates spatial and model columns from basin attributes, with notes on the Caravan feature families |
| `src/plots.py` | Exploratory plots (mostly commented out) |
