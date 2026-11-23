"""Forecast baseline for next week's Illinois cash bid.

We predict the change from this week's cash bid to next week's bid. The persistence benchmark
predicts a zero change. We compare it with two linear regressions:
  - cash only: the current cash bid and the prior weekly cash price change,
  - with basis: the same two inputs plus the current basis.

Here we only score the validation years 2020-2021, so the 2022-2025 test years stay untouched.
Each validation forecast comes from a regression refitted on an expanding window that only uses
outcomes known by the forecast date. All models are scored on the same weeks.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from config import PROCESSED, TABLES

CASH_ONLY = ["cash", "prev_change"]
WITH_BASIS = ["cash", "prev_change", "basis"]


def build_dataset():
    w = pd.read_csv(PROCESSED / "weekly_clean.csv", parse_dates=["date"])
    d = pd.DataFrame({
        "date": w["date"], "target_date": w["date"].shift(-1),
        "cash": w["cash"], "basis": w["basis"],
        "prev_change": w["cash"] - w["cash"].shift(),
        "y": w["cash"].shift(-1) - w["cash"],          # next week's cash change, what we forecast
        "period": w["period"],
    })
    # we need a seven-day gap on both sides, a valid contract label (for the basis), and no
    # suspected recording error in any of the three weeks involved
    gap_prev = (w["date"] - w["date"].shift()).dt.days
    gap_next = (w["date"].shift(-1) - w["date"]).dt.days
    ok_row = w["label_ok"] & ~w["suspect"]
    keep = (gap_prev == 7) & (gap_next == 7) & ok_row & ~w["suspect"].shift(-1, fill_value=False) \
        & ~w["suspect"].shift(1, fill_value=False)
    return d[keep].reset_index(drop=True)


def expanding_forecasts(d, cols, rows):
    preds = []
    for i in rows:
        known = d[d["target_date"] <= d.at[i, "date"]]
        fit = sm.OLS(known["y"], sm.add_constant(known[cols])).fit()
        x = np.r_[1.0, d.loc[i, cols].to_numpy(dtype=float)]
        preds.append(float(x @ fit.params.to_numpy()))
    return np.array(preds)


def dm_test(e_model, e_bench):
    """Diebold-Mariano test on absolute errors: is the model's error different from the benchmark's?"""
    dl = np.abs(e_model) - np.abs(e_bench)
    t = dl.mean() / (dl.std(ddof=1) / np.sqrt(len(dl)))
    return t, 2 * (1 - stats.norm.cdf(abs(t)))


def main(verbose=True):
    d = build_dataset()
    val = d.index[d["period"] == "validation"].to_numpy()
    y = 100 * d.loc[val, "y"].to_numpy()                       # cents per bushel
    preds = {
        "Persistence (no change)": np.zeros(len(val)),
        "Linear regression, cash only": 100 * expanding_forecasts(d, CASH_ONLY, val),
        "Linear regression, with basis": 100 * expanding_forecasts(d, WITH_BASIS, val),
    }
    rows = []
    for name, f in preds.items():
        e = y - f
        moved = y != 0
        # direction and the test only make sense for models that predict a change
        da = np.mean(np.sign(f[moved]) == np.sign(y[moved])) if name != "Persistence (no change)" else np.nan
        t, pv = dm_test(e, y) if name != "Persistence (no change)" else (np.nan, np.nan)
        rows.append({"model": name, "weeks": len(y), "MAE_c_bu": np.mean(np.abs(e)),
                     "RMSE_c_bu": np.sqrt(np.mean(e ** 2)), "directional_accuracy_pct": 100 * da,
                     "DM_stat_vs_persistence": t, "DM_p_value": pv})
    res = pd.DataFrame(rows)

    # coefficients fitted on the training years, to see what each input does
    train = d[d["period"] == "train"]
    coef = []
    for name, cols in [("cash only", CASH_ONLY), ("with basis", WITH_BASIS)]:
        fit = sm.OLS(100 * train["y"], sm.add_constant(train[cols])).fit()
        for k in fit.params.index:
            coef.append({"model": name, "term": k, "coef_c_bu": fit.params[k], "se": fit.bse[k],
                         "t": fit.tvalues[k], "r2": fit.rsquared, "n": int(fit.nobs)})
    coef = pd.DataFrame(coef)

    res.to_csv(TABLES / "table4_forecast_validation.csv", index=False)
    coef.to_csv(TABLES / "table5_forecast_coefficients_train.csv", index=False)
    if verbose:
        print(f"\nForecast sample: {len(d)} weeks (train {int((d['period'] == 'train').sum())}, "
              f"validation {len(val)}, test {int((d['period'] == 'test').sum())})")
        print("Table 4: forecast of next week's cash bid change, validation years 2020-2021 (cents/bu)")
        print(f"  {'model':<32s}{'weeks':>6s}{'MAE':>8s}{'RMSE':>8s}{'direction %':>13s}{'DM vs persistence':>20s}")
        for _, r in res.iterrows():
            dm = "" if np.isnan(r["DM_stat_vs_persistence"]) else f"{r['DM_stat_vs_persistence']:.2f} (p {r['DM_p_value']:.3f})"
            da = "" if np.isnan(r["directional_accuracy_pct"]) else f"{r['directional_accuracy_pct']:.1f}"
            print(f"  {r['model']:<32s}{r['weeks']:>6d}{r['MAE_c_bu']:>8.2f}{r['RMSE_c_bu']:>8.2f}{da:>13s}{dm:>20s}")
        print("Table 5: regression coefficients fitted on 2010-2019 (change in cents/bu; inputs in $/bu)")
        for _, r in coef.iterrows():
            print(f"  {r['model']:<11s} {r['term']:<12s} {r['coef_c_bu']:>9.4f} (SE {r['se']:.4f}, t {r['t']:.2f})  R2 {r['r2']:.4f}, n {r['n']}")
    return d, res


if __name__ == "__main__":
    main()
