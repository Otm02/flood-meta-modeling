# %%
import pathlib
from collections import defaultdict

import polars as pl
from rich.console import Console

console = Console()
use_mean = True
ft_mean_dir = pathlib.Path("data/finetuning_results_mean.csv")
ft_all_dir = pathlib.Path("data/finetuning_results_mean.csv")
ft = pl.read_csv(ft_mean_dir if use_mean else ft_all_dir)
console.print(f"Number of Columns: {len(ft.columns)}", style="red")

# %%


def flatten_dict(dd):
    fl = []
    for k, v in dd.items():
        fl.extend(v)
    return fl


meata_data_features = defaultdict(list)
meata_data_features["spatial"] = [
    "gauge_name",
    "country",
    "gauge_lat",
    "gauge_lon",
    "basin",
]
meata_data_features["km"] = [
    "area",
]

meata_data_features["model"] = [
    "",
    "val_delta",
    "test_origin",
    "val_origin",
    "test_delta",  # prediction output
]
meta_data_df = ft[flatten_dict(meata_data_features)]

"""
Note: Paper stated "Dataset limited to basins range of [100km^2, 2000km^2)"
    min value here is 93.5 (is it in km^2?)
"""

console.log(f"Area [100km^2, 2000km^2): {meta_data_df['area'].describe()}")


# %%
_ = """
# ERA5-Land meteorological variables
Feature (ERA5-Land variable name) Aggregation Unit
Precipitation (total_precipitation) Daily sum mm/day
Potential evaporation (potential_evaporation)i Daily sum mm/day
Air temperature (temperature_2m) Daily min/max and mean °C
Dew point temperature (dewpoint_temperature_2m) Daily min/max and mean °C
Shortwave radiation (surface_net_solar_radiation) Daily min/max and mean Wm−2
Net thermal radiation at the surface (surface_net_thermal_radiation) Daily min/max and mean Wm−2
Surface pressure (surface_pressure) Daily min/max and mean kPa
Eastward wind component (u_component_of_wind_10m) Daily min/max and mean ms−1
Northward wind component (v_component_of_wind_10m) Daily min/max and mean ms−1

# ERA5-Land model state variables
Feature (ERA5-Land variable name) Aggregation Unit
Snow water equivalent (snow_depth_water_equivalent) Daily min/max and mean mm
Soil water volume 0–7 cm (volumetric_soil_water_layer_1) Daily min/max and mean m3/m3
Soil water volume 7–28 cm (volumetric_soil_water_layer_2) Daily min/max and mean m3/m3
Soil water volume 28–100 cm (volumetric_soil_water_layer_3) Daily min/max and mean m3/m3
Soil water volume 100–289 cm (volumetric_soil_water_layer_4) Daily min/max and mean m3/m3
"""


# %%

"""
# HydroATLAS Dataset
## Hydrology Features: Description Aggregation Unit
Natural discharge (dis_m3_p[mn, mx,yr]) annual min/max/mean m3 s−1
Land surface runoff (run_mm_syr) spatial mean of sub-basin runoff mm
Inundation extent (inu_pc_s[mn, mx, lt]) annual min/mean and long-term max %
Limnicity - percent lake area (lka_pc_sse) spatial extent %
Lake Volume (lkv_mc_usu) at reach pour point 106 m3
Reservoir volume (rev_mc_usu) at reach pour point 106 m3
Degree of regulation (dor_pc_pva) index at reach pour point
River area (ria_ha_ssu) at reach pour point hectares
River volume (ria_tc_ssu) at reach pour point 103 m3
Groundwater table depth (gwt_cm_sav) spatial mean cm

## Physiography Features: Description Aggregation Unit
Elevation (ele_mt_s[av, mn, mx]) spatial mean/min/max m above sea level
Terrain slope (slp_dg_sav) spatial mean ° (x10)
Stream gradient (sgr_dk_sav) mean of reach segments dm/km


## Climate Features: Description Aggregation Unit
Climate zones from GEnS (clz_cl_smj) spatial majority classes (n=18)
Climate strata from GeNS (cls_cl_smj) spatial majority classes (n=125)
Air temperature (tmp_dc_s[01–12, mn, mx, yr]) monthly mean, annual mean/min/max °C (x10)
Precipitation (pre_mm_s[01–2, yr]) monthly mean, annual mean mm
Potential evapotranspiration (pet_mm_s[01–12, yr]) monthly mean, annual mean mm
Actual evapotranspiration (aet_mm_s[01–12, yr]) monthly mean, annual mean mm
Global aridity index (ari_ix_sav) spatial mean index value (x10)
Climate moisture index (cmi_ix_s[01–12, yr]) monthly mean, annual mean index value (x10)
Snow cover extent (snw_pc_s[01–12, mx, yr]) monthly mean, annual max/mean % cover


## Land Cover:
Land cover classes (glc_cl_smj) spatial majority classes (n=22)
Land cover extent (glc_pc_s[01–22]) spatial mean % cover
Potential natural vegetation classes (pnv_cl_smj) spatial majority classes (n=15)
Potential natural vegetation extent (pnv_pc_s[01–15]) spatial mean % cover
Wetland classes (wet_cl_smj) spatial majority classes (n=12)
Wetland extent (wet_pc_s[01–09, g1, g2]) spatial mean % cover & grouping
Forest cover extent (for_pc_sse) spatial mean % cover
Cropland extent (crp_pc_sse) spatial mean % cover
Pasture extent (pst_pc_sse) spatial mean % cover
Irrigated area extent (equipped) (ire_pc_sse) spatial mean % cover
Permafrost extent (prm_pc_sse) spatial mean % cover
Protected area extent (pac_pc_sse) spatial mean % cover
Terrestrial biomes (tbi_cl_smj) spatial majority classes (n=14)
Terrestrial ecoregions (tec_cl_smj) spatial majority classes (n=846)
Freshwater major habitat types (fmh_cl_smj) spatial majority classes (n=13)
Freshwater ecoregions (fec_cl_smj) spatial majority classes (n=426)


## Soils & Geology:
Clay fraction in soil (cly_pc_sav) spatial mean %
Silt fraction in soil (slt_pc_sav) spatial mean %
Sand fraction in soil (snd_pc_sav) spatial mean %
Organic carbon content in soil (soc_th_sav) spatial mean tonnes/hectare
Soil water content (swc_pc_s[01–12, yr]) monthly mean, annual mean %
Lithological classes (lit_cl_smj) spatial majority classes (n=16)
Karst area extent (kar_pc_sse) spatial mean % cover
Soil erosion (ero_kh_sav) spatial mean kg/hectare/yr


## Anthropogenic:
Population count (pop_ct_usu) at reach pour point count (thousands)
Population density (ppd_pk_sav) spatial mean people per km2
Urban extent (urb_pc_sse) spatial mean % cover
Nighttime lights (nli_ix_sav) spatial mean index value (x100)
Road density (rdd_mk_sav) spatial mean m/km2
Human footprint (hft_ix_s[93,09]) spatial mean for 1993 & 2009 index value (x100)
Gross domestic product (gdp_ud_sav) spatial mean USD ($)
Human development index (hdi_ix_sav) spatial mean index value (x1000)


## Climate attributes derived from ERA5-Land time series:
Attribute Description Unit
p_mean Mean daily precipitation mm/day
pet_mean Mean daily potential evaporation mm/day
aridity Aridity index, ratio of mean PET and mean precipitation
frac_snow Fraction of precipitation falling as snow
moisture_index Mean annual moisture index in range [−1, 1], where −1 indicates water-limited conditions and 1 energy-limited conditions
seasonality Moisture index seasonality in range [0, 2], where 0 indicates no changes in the water/energy
budget throughout the year and 2 indicates a change from fully arid to fully humid.
high_prec_freq Frequency of high precipitation days, where precipitation ≥5 times mean daily precipitation —
high_prec_dur Average duration of high precipitation events (number of consecutive days where precipitation ≥5 times mean daily precipitation days
low_prec_freq Frequency of low precipitation days, where precipitation <1 mmday-1 —
low_prec_dur Average duration of low precipitation events (number of consecutive days where precipitation <1 mmday-1 days
"""
hydrology_featuers = defaultdict(list)
physiography_features = defaultdict(list)
climate_features = defaultdict(list)
land_cover_features = defaultdict(list)
soil_geology_features = defaultdict(list)
anthropogenic_features = defaultdict(list)
era5_land_features = defaultdict(list)


for col in ft.columns:
    if "p_mean" in col or "pet_mean" in col:
        era5_land_features["mean"].append(col)
    elif "high_prec_dur" in col or "low_prec_dur" in col:
        era5_land_features["days"].append(col)
    elif (
        "aridity" in col
        or "frac_snow" in col
        or "high_prec_freq" in col
        or "low_prec_freq" in col
    ):
        era5_land_features["no_units"].append(col)
    elif "moisture_index" in col or "seasonality" in col:
        era5_land_features["index"].append(col)

    elif (
        "glc_cl" in col
        or "pnv_cl" in col
        or "wet_cl" in col
        or "tbi_cl" in col
        or "tec_cl" in col
        or "fmh_cl" in col
        or "fec_cl" in col
    ):
        land_cover_features["spatial_classes"].append(col)
    elif "glc_pc_s" in col or "wet_pc_s" in col or "pnv_pc_s" in col:
        if "wet_pc_sg1" in col or "wet_pc_sg2" in col:
            land_cover_features["spatial_percent"].append(col)
        else:
            land_cover_features["many_spatial_percent"].append(col)

    elif (
        "for_pc" in col
        or "pnv_pc_s" in col
        or "crp_pc" in col
        or "pst_pc" in col
        or "ire_pc" in col
        or "prm_pc" in col
        or "pac_pc" in col
    ):
        land_cover_features["spatial_percent"].append(col)

    elif "cly_pc" in col or "slt_pc" in col or "snd_pc" in col or "kar_pc" in col:
        soil_geology_features["percent"].append(col)
    elif "lit_cl_smj" in col:
        soil_geology_features["spatial_classes"].append(col)
    elif "ero_kh_sav" in col:
        soil_geology_features["erosion"].append(col)
    elif "soc_th_sav" in col:
        soil_geology_features["tonnes"].append(col)
    elif "swc_pc_s" in col:
        if "yr" in col:
            soil_geology_features["yearly_percent"].append(col)
        else:
            soil_geology_features["monthly_percent"].append(col)
    elif "urb_pc" in col:
        anthropogenic_features["percent"].append(col)
    elif "nli_ix_sav" in col or "hft_ix_s" in col or "hdi_ix_sav" in col:
        anthropogenic_features["index"].append(col)
    elif (
        "pop_ct_usu" in col
        or "ppd_pk_sav" in col
        or "rdd_mk_sav" in col
        or "gdp_ud_sav" in col
    ):
        anthropogenic_features["population"].append(col)

    elif "dis_" in col:
        hydrology_featuers["yearly"].append(col)
    elif "run_mm" in col or "gwt_cm" in col:
        hydrology_featuers["spatial"].append(col)
    elif "inu_" in col or "lka_" in col:
        hydrology_featuers["percent"].append(col)
    elif "lkv_" in col or "rev_" in col or "ria_" in col:
        # NOTE: Missing Column - River volume (ria_tc_ssu)
        hydrology_featuers["volume"].append(col)
    elif "dor_" in col:
        hydrology_featuers["degree"].append(col)
    elif "ele_mt_s" in col:
        physiography_features["elevation"].append(col)
    elif "slp" in col:
        physiography_features["terrain"].append(col)
    elif "sgr" in col:
        physiography_features["gradient"].append(col)
    elif "clz_cl" in col or "cls_cl" in col:
        # Zone + Strata Features
        # print(col)
        climate_features["spatial_classes"].append(col)

    # Monthly Mean climate features
    elif (
        "tmp_dc_s" in col
        or "pet_mm_s" in col
        or "pre_mm_s" in col
        or "cmi_ix_s" in col
        or "aet_mm_s" in col
        or "snw_pc_s" in col
    ):
        if "_syr" not in col:
            # Monthly mean climate features
            climate_features["monthly"].append(col)
        elif "_syr" in col:
            # Yearly mean climate features
            climate_features["yearly"].append(col)

    elif "ari_ix_sav" in col:
        climate_features["index"].append(col)

caravan_feature_set = [
    ("hydrology", ft[flatten_dict(hydrology_featuers)]),
    ("physiography", ft[flatten_dict(physiography_features)]),
    ("climate", ft[flatten_dict(climate_features)]),
    ("land_cover", ft[flatten_dict(land_cover_features)]),
    ("soil_geology", ft[flatten_dict(soil_geology_features)]),
    ("anthropogenic", ft[flatten_dict(anthropogenic_features)]),
    ("era5_land", ft[flatten_dict(era5_land_features)]),
]
total_caravan_cols = 0
for name, df in caravan_feature_set:
    df_col_len = len(df.columns)
    total_caravan_cols += df_col_len
    print(f"{name}: {df_col_len}")
console.print(f"Total Caravan Features: {total_caravan_cols}", style="red")
console.print(
    f"Expected Features Left: {len(ft.columns) - total_caravan_cols}", style="red"
)

dropping_cols: list[str] = []
for _, df in caravan_feature_set:
    dropping_cols += df.columns

dropping_cols = dropping_cols + meta_data_df.columns
un_accounted_for_cols = set(ft.columns) - set(dropping_cols)
print(un_accounted_for_cols)

# %%
usable_features_for_training = []
usable_features_for_training += meata_data_features["km"]

for k, v in hydrology_featuers.items():
    usable_features_for_training += v

for k, v in physiography_features.items():
    usable_features_for_training += v

for k, v in climate_features.items():
    if k != "monthly":
        usable_features_for_training += v
for k, v in land_cover_features.items():
    if k != "many_spatial_percent":
        usable_features_for_training += v

for k, v in soil_geology_features.items():
    if k != "montly_percent":
        usable_features_for_training += v

for k, v in era5_land_features.items():
    if k == "no_units":
        usable_features_for_training += v


# %%
console.print(f"Usable Feature: {len(usable_features_for_training)}", style="red")
console.print(f"Usable Feature: {usable_features_for_training}", style="red")
