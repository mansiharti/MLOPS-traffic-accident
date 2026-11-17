# Data Validation Rules — Traffic Accident Severity Analytics

Implemented in `scripts/clean.py`. Every accident row is checked against every rule below.
A row that fails **any** rule is written to the rejected layer with the reason(s) recorded.

| Rule name | Description | Why it matters |
|---|---|---|
| `missing_accident_id` | `accident_index` must not be blank | The ID is the primary key — without it we can't reliably identify or deduplicate the record |
| `missing_date_or_time` | `accident_date` AND `accident_time` must both be present | Nearly every dashboard view (hour, weekday, date trends) depends on this |
| `invalid_severity` | `accident_severity` must be exactly one of `Fatal`, `Serious`, `Slight` | Anything else (blank, "Unknown", typos) would break severity-based charts and SQL `CHECK` constraints |
| `invalid_coordinates` | `latitude`/`longitude` must both be present, numeric, not `(0,0)`, and inside the UK bounding box (lat 49.8–60.9, lon -8.6–1.8) | Bad GPS values would break the hotspot map and any location-based weather lookup |
| `missing_required_field` | `road_type`, `number_of_vehicles`, `number_of_casualties` must all be present | These are used directly in dashboard views and the analytics mart |
| `duplicate_accident_id` | No two rows may share the same non-blank `accident_index` | The same accident should not be double-counted in any aggregate |

## Design notes

- Checks run **independently** per row (except the duplicate check, which needs the whole table)
  so each one is easy to read, test, and explain on its own.
- A row can fail **multiple** rules at once — all applicable reasons are recorded together
  (comma-separated) in `rejection_reasons`, not just the first one found.
- Rejected rows are **never deleted** — they are kept in `data/rejected/accidents_rejected.csv`
  and later loaded into the `rejected_accidents` Postgres table, so they can be audited later.
- No row is ever silently "fixed" or guessed at — a row either fully passes every check, or it is
  rejected with a clear reason. This avoids introducing fabricated/estimated values into the
  cleaned data.
