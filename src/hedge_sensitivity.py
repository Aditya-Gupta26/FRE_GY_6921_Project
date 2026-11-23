"""Holding-period and estimation-window checks for the existing corn hedge.

Run from the repository root:
    python src/hedge_sensitivity.py
    python src/hedge_sensitivity.py --horizons 1 2 4 --windows 52 104 156

The hedge ratio is always estimated from completed eligible ONE-WEEK changes,
as in hedge_baseline.py. The position is then held fixed for the chosen horizon.
Windows count eligible observations, not calendar weeks.

Each holding interval must be a chain of existing eligible weekly pairs with
one contract throughout. Test entries follow a fixed nonoverlapping calendar
anchored on the first Friday of the test period. Within a horizon, all windows
and strategies use identical dates. Different horizons have different samples.

These are descriptive checks on the historical, cleaned sample. The source
only supplies a nearby quote, and its roll exclusions and suspected-error flags
are retrospective. The grid does not select a new model using test performance.
"""
import argparse

import numpy as np
import pandas as pd

from config import CONTRACT_BU, FULL_HEDGE, PROCESSED, SALE_BU, TABLES, VALID_LAST_YEAR
from hedge_baseline import contracts, load_pairs, ols_slope


def holding_pairs(pairs, horizon):
    """Join complete chains of eligible weekly pairs, without crossing a roll or gap."""
    valid = pd.Series(True, index=pairs.index)
    for offset in range(horizon):
        expected = pairs["entry_date"] + pd.Timedelta(weeks=offset)
        valid &= pairs["entry_date"].shift(-offset).eq(expected)
        valid &= pairs["contract"].shift(-offset).eq(pairs["contract"])

    holds = pairs[["entry_date", "contract", "fnd", "cash_entry", "fut_entry"]].copy()
    last = pairs[["sale_date", "cash_sale", "fut_sale"]].shift(-(horizon - 1))
    holds = pd.concat([holds, last], axis=1).loc[valid].reset_index(drop=True)
    holds["dS"] = 100 * (holds["cash_sale"] - holds["cash_entry"])
    holds["dF"] = 100 * (holds["fut_sale"] - holds["fut_entry"])
    return holds


def rolling_estimates(pairs, entries, window):
    """Fit each ratio using the last window pairs whose sale is known at entry."""
    rows = []
    for entry in entries:
        history = pairs.loc[pairs["sale_date"] <= entry].tail(window)
        row = {"entry_date": entry, "h": np.nan, "estimate_start": pd.NaT,
               "estimate_end": pd.NaT, "estimate_pairs": len(history)}
        if len(history) == window and history["dF"].var(ddof=1) > 0:
            row["h"] = ols_slope(history["dS"].to_numpy(), history["dF"].to_numpy())[0]
            row["estimate_start"] = history["entry_date"].iloc[0]
            row["estimate_end"] = history["sale_date"].iloc[-1]
        rows.append(row)
    return pd.DataFrame(rows).set_index("entry_date")


def summarize(ledger):
    """Compare variance of holding-period price changes on matched observations."""
    rows = []
    for (horizon, window), group in ledger.groupby(["horizon_weeks", "window_observations"]):
        unhedged = group.loc[group["strategy"] == "no_hedge", "dV_c_bu"]
        full = group.loc[group["strategy"] == "one_to_one", "dV_c_bu"]
        variance = unhedged.var(ddof=1)
        if len(unhedged) < 2 or variance <= 0:
            raise ValueError(f"Horizon {horizon} needs at least two trades with nonzero cash variance.")
        full_variance = full.var(ddof=1)
        for strategy, trades in group.groupby("strategy", sort=False):
            change = trades["dV_c_bu"]
            reduction = 100 * (1 - change.var(ddof=1) / variance)
            rows.append({
                "horizon_weeks": horizon, "window_observations": window,
                "strategy": strategy, "trades": len(trades),
                "first_entry": trades["entry_date"].min(), "last_sale": trades["sale_date"].max(),
                "mean_dV_c_bu": change.mean(), "sd_dV_c_bu": change.std(ddof=1),
                "variance_reduction_pct": reduction,
                "vr_minus_1to1_pp": 100 * (full_variance - change.var(ddof=1)) / variance,
                "worst_trade_c_bu": change.min(),
                "sd_proceeds_change_usd": change.std(ddof=1) * SALE_BU / 100,
                "contracts_min": trades["contracts"].min(),
                "contracts_mean": trades["contracts"].mean(),
                "contracts_max": trades["contracts"].max(),
            })
    return pd.DataFrame(rows)


def run_sensitivity(pairs, horizons=(1, 2, 4), windows=(52, 104, 156)):
    """Return a summary, an auditable trade ledger, and sample-coverage counts."""
    if not horizons or any(h < 1 or int(h) != h for h in horizons):
        raise ValueError("Horizons must be positive whole weeks.")
    if not windows or any(w < 3 or int(w) != w for w in windows):
        raise ValueError("Estimation windows must contain at least three observations.")
    horizons = sorted({int(horizon) for horizon in horizons})
    windows = sorted({int(window) for window in windows})
    pairs = pairs.sort_values("entry_date").reset_index(drop=True)
    training = pairs.loc[pairs["period"] == "train"]
    if len(training) < 3 or training["dF"].var(ddof=1) <= 0:
        raise ValueError("The static benchmark needs at least three training pairs and nonzero futures variance.")
    static_ratio = ols_slope(training["dS"].to_numpy(), training["dF"].to_numpy())[0]
    static_count = contracts(static_ratio)
    test_start = pd.Timestamp(VALID_LAST_YEAR + 1, 1, 1)
    anchor = test_start + pd.Timedelta(days=(4 - test_start.weekday()) % 7)

    ledgers, coverage = [], []
    for horizon in horizons:
        holds = holding_pairs(pairs, horizon)
        test = holds.loc[holds["entry_date"] >= test_start].copy()
        on_schedule = (test["entry_date"] - anchor).dt.days.mod(7 * horizon).eq(0)
        scheduled = test.loc[on_schedule].set_index("entry_date", drop=False)
        if scheduled.empty:
            raise ValueError(f"No eligible scheduled test trades for horizon {horizon}.")
        estimates = {window: rolling_estimates(pairs, scheduled.index, window) for window in windows}
        common = pd.Series(True, index=scheduled.index)
        for estimate in estimates.values():
            common &= np.isfinite(estimate["h"])
        selected = scheduled.loc[common]
        if len(selected) < 2:
            raise ValueError(
                f"Horizon {horizon} has fewer than two common test trades for windows {windows}. "
                "Use shorter windows or a shorter holding period."
            )
        if training["sale_date"].max() > selected["entry_date"].min():
            raise ValueError("Static training observations must end before the first test entry.")
        for window, estimate in estimates.items():
            fitted = estimate.loc[selected.index]
            coverage.append({
                "horizon_weeks": horizon, "window_observations": window,
                "eligible_test_holds": len(test), "scheduled_test_holds": len(scheduled),
                "holds_with_full_window": int(np.isfinite(estimate["h"]).sum()),
                "common_test_trades": len(selected),
            })
            rolling_count = fitted["h"].map(contracts)
            positions = {"no_hedge": 0, "one_to_one": FULL_HEDGE,
                         "static_min_variance": static_count, "rolling_min_variance": rolling_count}
            for strategy, count in positions.items():
                trades = selected.copy()
                trades["horizon_weeks"] = horizon
                trades["window_observations"] = window
                trades["strategy"] = strategy
                trades["contracts"] = count
                trades["estimate_start"] = pd.NaT
                trades["estimate_end"] = pd.NaT
                trades["estimate_pairs"] = 0
                trades["h_estimated"] = np.nan
                if strategy == "rolling_min_variance":
                    trades[["estimate_start", "estimate_end", "estimate_pairs"]] = (
                        fitted[["estimate_start", "estimate_end", "estimate_pairs"]]
                    )
                    trades["h_estimated"] = fitted["h"]
                elif strategy == "static_min_variance":
                    trades["estimate_start"] = training["entry_date"].min()
                    trades["estimate_end"] = training["sale_date"].max()
                    trades["estimate_pairs"] = len(training)
                    trades["h_estimated"] = static_ratio
                trades["dV_c_bu"] = trades["dS"] - CONTRACT_BU / SALE_BU * trades["contracts"] * trades["dF"]
                trades["futures_pnl_usd"] = CONTRACT_BU * trades["contracts"] * (trades["fut_entry"] - trades["fut_sale"])
                trades["sale_proceeds_usd"] = SALE_BU * trades["cash_sale"] + trades["futures_pnl_usd"]
                trades["proceeds_change_usd"] = SALE_BU / 100 * trades["dV_c_bu"]
                ledgers.append(trades.reset_index(drop=True))
    ledger = pd.concat(ledgers, ignore_index=True)
    return summarize(ledger), ledger, pd.DataFrame(coverage)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 4], help="Holding periods in whole weeks.")
    parser.add_argument("--windows", nargs="+", type=int, default=[52, 104, 156], help="Completed eligible weekly observations used to estimate each ratio.")
    args = parser.parse_args()
    try:
        summary, ledger, coverage = run_sensitivity(load_pairs(), args.horizons, args.windows)
    except ValueError as error:
        parser.error(str(error))
    TABLES.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLES / "hedge_sensitivity.csv", index=False, date_format="%Y-%m-%d")
    coverage.to_csv(TABLES / "hedge_sensitivity_coverage.csv", index=False)
    ledger.to_csv(PROCESSED / "hedge_sensitivity_ledger.csv", index=False, date_format="%Y-%m-%d")
    columns = ["horizon_weeks", "window_observations", "strategy", "trades",
               "sd_dV_c_bu", "variance_reduction_pct", "vr_minus_1to1_pp"]
    print(summary[columns].to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print("\nWindows use completed one-week observations. Positions stay fixed until sale.")
    print("Entry dates do not overlap within each horizon; samples differ between horizons.")
    print("Results are descriptive sensitivity checks, with the baseline cleaning and calendar limitations.")
    print(f"Saved summary and coverage to {TABLES}; trade ledger to {PROCESSED}.")
    return summary, ledger, coverage


if __name__ == "__main__":
    main()
