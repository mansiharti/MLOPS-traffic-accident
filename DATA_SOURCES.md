# Dataset Sources and Access

## Accident records

The raw input is the UK Department for Transport Road Safety Data, obtained from the
[UK Government data portal](https://www.data.gov.uk/dataset/road-accidents-safety-data).
It is distributed under the Open Government Licence v3.0. The project copy is stored
under `data/raw/uk_2025/` as three related CSV datasets:

- `collisions.csv` contains one row per collision and the main accident attributes.
- `casualties.csv` contains casualty-level records joined by `collision_index`.
- `vehicles.csv` contains vehicle-level records joined by `collision_index`.

The pipeline aggregates the casualty and vehicle detail rows to collision-level counts,
using the collision file's counts only as a fallback when a detail row is unavailable.

To use a refreshed extract, download the relevant accidents CSV, place it in
`data/raw/`, and run `python main.py`. The pipeline selects the first CSV in that
directory, so keep only one input extract there for a deterministic run.

## Weather enrichment

The current enrichment stage uses the DfT collision dataset's recorded
`weather_conditions` field and maps it to descriptive weather values. The generated
weather fields are written to `data/output/accidents_enriched.csv`.

`data/output/weather_cache.json` records the weather source metadata for reproducibility;
the current implementation does not make live Open-Meteo requests.

## Access and reproducibility

No API key is required. An internet connection is needed only for uncached Open-Meteo
requests. The warehouse can use PostgreSQL when the variables in `.env` are configured;
otherwise the default and submission-ready target is the local SQLite database at
`data/traffic_accidents.db`.