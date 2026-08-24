# Data provenance — monthly case dataset (2023–2025)

The app's earlier dataset was **real** RDT-confirmed case data, but only at
**annual** resolution — not enough to test a monthly weather/case relationship
or drive a monthly-cadence forecast. With the project supervisor's approval,
that annual dataset was replaced with a **simulated** monthly dataset
(`zambia_monthly_malaria_simulated.csv`) covering 106 districts, Jan 2023
through Dec 2025.

This is disclosed persistently in the running app itself — a badge in the
sidebar and a banner at the top of every page — not just here.

## What's real vs. simulated in the current dataset

| Field | Status |
|---|---|
| Confirmed case counts, dates | **Simulated** (from `zambia_monthly_malaria_simulated.csv`) |
| District coordinates, province | Real (Wikidata-sourced, unchanged by this swap) |
| District population | Real (2022 census, unchanged by this swap) |
| Rainfall / temperature | Real historical weather (Open-Meteo ERA5 archive), fetched per district for the exact 2023–2025 window this simulated dataset covers |

## How it was produced

1. `preprocess_simulated.py` — reads the source CSV (`Year, Month, Province,
   District, Suspected_Cases, RDT_Tested, Microscopy_Tested, Confirmed_Cases`),
   fixes 3 district-name spelling variants so they match the app's existing
   real district records instead of creating duplicates
   (`Itezhi-Tezhi`→`Itezhi-tezhi`, `Mushindamo`→`Mushindano`,
   `Shang'ombo`→`Shangombo`), and writes a CSV in the exact shape the app's
   own Upload Data page expects (`district, date, epi_week, reporting_year,
   rdt_confirmations, suspected_cases, rdt_tested, microscopy_tested`), with
   `date` set to the 1st of each month.
2. That CSV was loaded through the app's real upload pipeline (same code
   path as the Upload Data page), not a separate import route. It was run
   twice: once for the core fields, and again after `suspected_cases`,
   `rdt_tested`, and `microscopy_tested` were added to the schema — the
   second pass updated the same 3,816 records in place (matched on
   district + date) without touching the already-backfilled weather
   fields, since those aren't part of what the upload form sets.

`rdt_tested` (test volume) alongside `rdt_confirmations` (confirmed cases)
means a real positivity rate (confirmed ÷ tested) can now be computed per
district per month — not yet surfaced in the UI, just available in the data.
3. `fetch_weather.py` — for each of the 106 districts, fetches daily
   rainfall and mean temperature from Open-Meteo's historical archive API
   for the district's real coordinates over 2023-01-01–2025-12-31, then
   aggregates to monthly rainfall (sum) and monthly average temperature
   (mean), and writes those values onto the matching records directly.

Before any of this, the previous 580 real annual records were exported to
a backup CSV (kept outside the repo) so the swap is reversible if needed.

## A note on the old parser bug this surfaced

Dry-running the source CSV through the existing Upload Data parser (before
any preprocessing) surfaced two real bugs in the app's automatic
column-detection, since it was never tested against a Year+Month (no
combined `date` column) file before:

- No `date` column exists, so the parser fell back to matching the `Month`
  column as if it were a date, which fails to parse and would have
  collapsed every row for a district onto today's date (silently
  overwriting all but the last row per district).
- The column-matching keyword `rdt` matched `RDT_Tested` (test volume)
  before `confirm` matched `Confirmed_Cases` (actual case count), since
  `RDT_Tested` appears earlier in the file's column order.

`preprocess_simulated.py` works around the missing-`date`-column issue by
producing an explicit `YYYY-MM-01` date. The confirmed-vs-tested column
ambiguity was fixed properly in the live parser itself (`upload_view` in
`views.py`) once `rdt_tested` became a real field it needed to detect
anyway — it now checks column headers for `confirm` across the whole file
before ever falling back to the more generic `rdt`, instead of just
matching whichever relevant-looking column happens to appear first.
