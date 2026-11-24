# Hedging Corn Price Risk with CBOT Futures and Forecasting Illinois Cash Bids

FRE-GY 6921 Commodities: Markets and Derivatives, Group 4
Luanqi Tong, Aditya Gupta, Elif Erdogan, Yuan Zou
Repository: https://github.com/Aditya-Gupta26/FRE_GY_6921_Project

An Illinois corn grower expects to sell 50,000 bushels to an elevator in one week. We test whether a
rolling minimum-variance hedge with CBOT corn futures (ZC) stabilizes the grower's effective sale price
more than a one-for-one hedge of 10 contracts. We also test whether the current local basis improves a
one-week-ahead forecast of the Illinois cash bid.

This repository currently holds our working baseline: the cleaned data extract, the hedge comparison,
and a first forecast comparison against the persistence benchmark.

## Setup

We tested the code with Python 3.13 (pandas 3.1, NumPy 2.5) and with pandas 2.2 / NumPy 2.1.

```
pip install -r requirements.txt
python src/run_baseline.py
```

The run takes a few seconds. It uses the USDA snapshot saved in `data/raw/`, so the numbers match ours
exactly. To pull a fresh copy from USDA first, run `python src/run_baseline.py --download`. USDA adds a
new week every Friday, but our analysis window is fixed, so new weeks do not change the results unless
USDA revises older weeks.

## Data

We use the Illinois Elevator Bid series for corn from the weekly Grain Basis dataset of the USDA
Agricultural Marketing Service (https://agtransport.usda.gov/Grain/Grain-Basis/v85y-3hep). Each week
has the cash bid, the nearby CBOT futures price, the basis (cash bid minus futures price) and the
futures contract month, all in U.S. dollars per bushel.

`src/download_data.py` pulls the series through the dataset's public API with the filter
`market_name='Illinois' AND market_type='Elevator Bid' AND commodity='Corn'`. Our snapshot was taken
on November 21, 2026 and has 1,008 weeks from January 5, 2007 to November 13, 2026
(`data/raw/retrieval_info.json`). We study January 8, 2010 to December 26, 2025, which gives 811
weeks, the same count as in our initial proposal.

For the market context we also pull the last 52 weekly CFTC Commitments of Traders reports for CBOT
corn (disaggregated, futures only, contract code 002602) from the CFTC public reporting API
(https://publicreporting.cftc.gov/Commitments-of-Traders/Disaggregated-Futures-Only/72hh-3qpy). Our
snapshot's latest report is for positions on November 17, 2026.

## How we clean the data

- We infer each futures quote's delivery year from the quote date and contract month (a March label
  quoted in November is next year's March contract) and check it against the CBOT corn calendar.
  Four weeks fail: one nonstandard January label, and three weeks that quote a March or December
  contract after its last trading day.
- One week, June 3, 2016, has a cash bid of 4.68 between 3.94 and 4.04, while futures barely move and
  the basis jumps from -0.19 to +0.50 and back. We treat it as a recording error and exclude it. The
  rule is in `suspected_errors()` in `src/clean_data.py`.
- We form pairs of weeks exactly seven days apart. A pair enters the hedge sample only if both weeks
  quote the same futures contract and the sale happens before first notice day, so the grower is never
  holding futures in the delivery period.
- `data/processed/eligibility_log.csv` lists every pair and the reason it was kept or excluded.

## Folder structure

```
data/raw/          USDA and CFTC snapshots and pull details
data/processed/    cleaned weekly data, eligibility log, weekly hedge ledger for the test years
src/               Python code (run_baseline.py runs everything)
results/tables/    tables as CSV
results/figures/   charts as PNG
results/baseline_summary.txt   printed output of the full run
excel/             Excel checks of the proceeds calculation
docs/              project documents
```

## Where each result comes from

| Result | File | Code |
|---|---|---|
| Table 1, sample funnel | `results/tables/table1_sample_funnel.csv` | `src/clean_data.py` |
| Table 2, hedge ratio and contracts | `results/tables/table2_hedge_ratio.csv` | `src/hedge_baseline.py` |
| Table 3, hedge comparison (train and test) | `results/tables/table3_hedge_performance.csv` | `src/hedge_baseline.py` |
| Table 4, forecast comparison (validation) | `results/tables/table4_forecast_validation.csv` | `src/forecast_baseline.py` |
| Table 5, regression coefficients (train) | `results/tables/table5_forecast_coefficients_train.csv` | `src/forecast_baseline.py` |
| Figure 1, cash bid, futures and basis | `results/figures/figure1_prices_and_basis.png` | `src/make_figures.py` |
| Figure 2, hedge regression (train) | `results/figures/figure2_hedge_regression_train.png` | `src/make_figures.py` |
| Figure 3, contracts held in the test years | `results/figures/figure3_rolling_hedge_size_test.png` | `src/make_figures.py` |
| Weekly hedge ledger, test years | `data/processed/hedge_test_weekly_ledger.csv` | `src/hedge_baseline.py` |
| Market snapshot (latest basis, COT positions) | `results/tables/table6_market_snapshot.csv` | `src/market_snapshot.py` |

The table and figure numbers in our revised proposal (`docs/FRE6921_Group04_RevisedProposal.docx`) differ
from the file names: proposal Table 3 is the sample funnel, Table 4 the hedge comparison, Table 5 the
forecast comparison, and Figures 1 and 2 are the first two figure files above.

## Sample split

We train on 2010-2019, select model settings on 2020-2021, and keep 2022-2025 for the out-of-sample
test. The hedge comparison is reported on the training years and on the test years. The rolling hedge
ratio uses only weeks whose sale date is on or before the hedge date. The forecast models are scored
on the validation years only, so the test years stay untouched until the final report.

## Hedge sensitivity checks

Luanqi's contribution runs independently of the baseline:

```
python src/hedge_sensitivity.py
python src/hedge_sensitivity.py --horizons 1 2 4 --windows 52 104 156
```

It compares no hedge, the 10-contract hedge, the static hedge estimated on the training years, and
the rolling hedge. Every rolling ratio uses completed eligible **one-week** changes. The position
is fixed at entry for the chosen holding period. A window of 104 means 104 eligible observations,
which can span more than 104 calendar weeks.

A holding period must be a complete chain of eligible weekly pairs with the same delivery contract
throughout. Entry dates follow a fixed Friday schedule beginning January 7, 2022, spaced by the
holding period. This gives nonoverlapping trades within each horizon. Within a horizon, all windows
and all strategies use the same entry dates. Longer horizons have different samples, so differences
across horizons also reflect which dates remain eligible.

The script writes three new files:

| File | Contents |
|---|---|
| `results/tables/hedge_sensitivity.csv` | Risk measures for each horizon, window and strategy |
| `results/tables/hedge_sensitivity_coverage.csv` | Candidate, scheduled and retained trade counts |
| `data/processed/hedge_sensitivity_ledger.csv` | Positions, estimation dates, proceeds and changes for every trade |

With the saved data and the 104-observation window, the results are:

| Holding period | Test trades | Variance reduction for 1:1 | Variance reduction for rolling |
|---|---:|---:|---:|
| 1 week | 168 | 72.53% | 73.01% |
| 2 weeks | 72 | 75.12% | 75.22% |
| 4 weeks | 25 | 56.76% | 54.78% |

Variance reduction measures the change in effective sale price relative to its entry cash value.
The `vr_minus_1to1_pp` column is the difference in variance reduction in percentage points.
The one-week, 104-observation result reproduces the existing hedge ledger. The four-week sample
has only 25 trades. Treat the grid as descriptive sensitivity and retain the prespecified
104-observation baseline when discussing the main results.

The analysis inherits the existing cleaning choices, retrospective roll exclusions, and exchange
calendar that skips weekends but does not adjust for exchange holidays. It excludes transaction
and margin financing costs, as the baseline does.
