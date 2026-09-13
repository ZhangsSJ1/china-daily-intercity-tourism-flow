# Reconstructing Daily Intercity Tourism Flows in China

This repository provides a Python workflow for reconstructing daily intercity tourism flows across China from 2012 to 2021. The study covers 339 prefecture-level and above administrative units. It combines city-to-city travel links extracted from online travel blogs with annual domestic tourist arrivals, tourism points of interest (POIs), socioeconomic indicators, weather records, and calendar information.

The workflow produces city-level daily tourism arrivals and annual files of estimated directed origin-destination (OD) tourism flows, calibrated against annual city-level domestic tourism totals.

## Workflow

The pipeline consists of three consecutive stages.

1. **Annual domestic tourism estimation.** A log-linear ordinary least squares model with province-year fixed effects is used to complete missing annual domestic tourist arrivals. Observed statistical records are retained, while missing values are estimated from tourism POIs and corrected using the Duan smearing estimator.

2. **Daily city-level tourism estimation.** A LightGBM model learns daily variation in destination-city travel-blog counts from weather, calendar structure, city attributes, and city-specific monthly seasonality. The predicted daily values are then normalized within each city-year so that their annual sum equals the annual domestic tourism total.

3. **Daily intercity OD flow allocation.** A second LightGBM model learns relative OD weights from travel-blog OD records using origin push factors, destination pull factors, spatial resistance, weather, and calendar features. Predicted OD weights are normalized by destination and date so that total inbound flow equals the estimated daily tourism arrivals of each destination city.

The outputs are model-based estimates. Travel blogs provide information on relative intercity connectivity and temporal variation, whereas annual domestic tourism statistics provide the volume constraint.

## Repository Structure

```text
.
├── full_pipeline_tourism_od.py
└── README.md
```

All output files are written to the directory specified by `DATA_DIR` at the beginning of the script.

## Requirements

Python 3.9 or later is recommended.

```bash
pip install pandas numpy statsmodels lightgbm openpyxl
```

Before running the script, update `DATA_DIR` to the local directory containing the input data. The same directory is used for generated outputs.

```python
DATA_DIR = Path(r"D:\\your_data_directory")
OUTPUT_DIR = DATA_DIR
```

Run the workflow with:

```bash
python full_pipeline_tourism_od.py
```

## Input Data

The following files are required. City names, dates, years, and the ordering of the distance matrix must be harmonized before execution.

| File | Key fields or worksheet | Purpose |
|---|---|---|
| `城市维表.csv` | `city_name`, `city_code_ref` | Defines the study cities and their order. |
| `国内旅游人数.xlsx` | `Sheet2`; `省份`, `城市`, `2012年`-`2021年` | Annual domestic tourist arrivals by city. |
| `city_year_tourism_poi_counts.csv` | `city`, `province`, `year`, `uniq10_n`, and POI counts | Annual tourism-resource endowment. |
| `Edges_City_By_Departure_Date.csv` | `Flow_Date`, `Source_City_ZH`, `Target_City_ZH`, `Trip_Count`, `Trip_Normalized_Weight` | Daily intercity links derived from travel blogs and model supervision data. |
| `逐日日历_2012-2021.csv` | `flow_date` and calendar/holiday fields | Calendar and holiday features. |
| `city_distance_matrix_339.csv` | Square matrix indexed by city names | Intercity distance matrix. |
| `中国城市数据面板数据（2000-2024年）.xlsx` | `线性插值`; year, city, province, GDP, tertiary industry, registered population, GDP per capita | City-level socioeconomic features. |
| `城市逐日气候_2012.csv` to `城市逐日气候_2021.csv` | `flow_date`, `city_name`, `tmean_c`, `precip_mm` | Daily city-level weather features. |

The script matches records by city name rather than administrative code.

## Output Data

| File | Description |
|---|---|
| `annual_tourism_panel_339x10.csv` | Annual domestic tourism panel for 339 cities from 2012 to 2021. |
| `ols_model_summary.txt` | Summary of the annual tourism estimation model. |
| `daily_tourism_panel_339_v2.csv` | Daily city-level tourism arrivals. |
| `step3_model_metrics.csv`, `step3_feature_importance.csv` | Performance records and feature importance for the daily city-level model. |
| `od_tourism_flow_2012.csv` to `od_tourism_flow_2021.csv` | Annual files containing daily directed intercity OD tourism-flow estimates. |
| `step4_model_metrics.csv`, `od_flow_feature_importance.csv` | Performance records and feature importance for the OD model. |
| `od_flow_lgbm_model.txt` | Trained OD LightGBM model. |
| `feature_selection_doc.txt` | Description of the OD feature system. |

Each OD flow file contains four principal fields:

- `origin`: origin city;
- `dest`: destination city;
- `date`: date indexed by the departure date of the travel blog;
- `flow_10k`: estimated tourism flow, in units of 10,000 person-trips.

## Runtime Notes

- When `SKIP_PREDICT_IF_EXISTS = True` and all annual OD files already exist, the script skips the full OD prediction stage to avoid overwriting large result files. The preceding model-training and model-information export stages are still executed.
- Setting `SKIP_PREDICT_IF_EXISTS = False` regenerates all OD files from 2012 to 2021. Ensure that adequate disk space is available and that existing outputs can be replaced.
- Both LightGBM models use a fixed random seed of 42. The daily city-level model is split by city, and the OD model is split by directed city pair for training and validation.

## Data Definitions and Limitations

- All intercity transitions within one travel blog are assigned to the blog's `Departure_Date`. Therefore, `date` is a departure-date indexing variable rather than the actual arrival date of every travel segment.
- Annual domestic tourist arrivals and intercity inbound tourism flows do not have identical statistical definitions. Annual arrivals are used as a destination-level volume constraint; the resulting OD values should be interpreted as annual-total-calibrated estimates of intercity tourism flows.
- Unrecorded OD destinations may be sampled as zero-label observations during model training. These zero labels do not indicate that tourism travel did not occur in the real world.
- The output is intended to characterize relative spatial patterns, flow directions, and daily variation. Interpretations of absolute volumes should consider travel-blog coverage, sharing behavior, and differences in statistical definitions.

## Data Availability and Reuse

Raw travel blogs, POIs, weather records, statistical yearbook data, and other source data are not included in this repository because their distribution may be subject to platform terms, licenses, or other access restrictions. When using this workflow or derived data, please describe the data sources, study units, study period, annual volume constraint, and departure-date aggregation rule.
