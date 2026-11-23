"""Baseline charts, saved to results/figures/.

Figure 1 shows the cash bid, futures price and basis over the whole sample, Figure 2 the training
regression behind the hedge ratio, and Figure 3 how many contracts each hedge holds in the test years.
The figure numbers and titles go in the captions of our write-up, so the images carry no titles.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import FIGURES, FULL_HEDGE, PROCESSED, ROLL_WINDOW, TABLES

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PERIODS = [("2010-01-01", "2020-01-01", "Train 2010-2019", "#f3f2ef"),
           ("2020-01-01", "2022-01-01", "Validation\n2020-2021", "#e9e8e4"),
           ("2022-01-01", "2026-01-01", "Test 2022-2025", "#f3f2ef")]

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def shade(ax, label=False):
    """Shade the train, validation and test years in the background."""
    for a, b, name, col in PERIODS:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=col, zorder=0, lw=0)
        if label:
            ax.text(pd.Timestamp(a) + (pd.Timestamp(b) - pd.Timestamp(a)) / 2, 1.0, name, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=7.5, color=MUTED)


def figure1(w):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 4.7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    shade(a1, label=True)
    shade(a2)
    a1.plot(w["date"], w["cash"], color=BLUE, lw=1.2, label="Illinois elevator bid (cash)")
    a1.plot(w["date"], w["futures"], color=ORANGE, lw=1.2, label="CBOT corn nearby futures")
    a1.set_ylabel("Price ($/bu)")
    a1.legend(loc="upper left", bbox_to_anchor=(0.27, 0.98))
    a2.axhline(0, color=MUTED, lw=0.8)
    a2.plot(w["date"], w["basis"], color=AQUA, lw=1.2)
    s = w[w["suspect"]]
    a2.scatter(s["date"], s["basis"], marker="x", s=40, color=INK, zorder=3)
    for _, r in s.iterrows():
        a2.annotate("excluded: suspected\nrecording error", (r["date"], r["basis"]), xytext=(12, 4),
                    textcoords="offset points", fontsize=8, color=INK)
    a2.set_ylabel("Basis ($/bu)")
    a2.set_xlabel("Week ending date")
    a2.xaxis.set_major_locator(mdates.YearLocator(2))
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(FIGURES / "figure1_prices_and_basis.png", dpi=200)
    plt.close(fig)


def figure2(p):
    tr = p[p["eligible"] & (p["period"] == "train")]
    ds, df = 100 * (tr["cash_sale"] - tr["cash_entry"]), 100 * (tr["fut_sale"] - tr["fut_entry"])
    fit = sm.OLS(ds, sm.add_constant(df)).fit()
    lim = np.ceil(max(ds.abs().max(), df.abs().max()) / 10) * 10
    x = np.array([-lim, lim])
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.scatter(df, ds, s=14, color=BLUE, alpha=0.55, edgecolor="none", label=f"Eligible weekly pairs (n = {int(fit.nobs)})")
    ax.plot(x, fit.params.iloc[0] + fit.params.iloc[1] * x, color=ORANGE, lw=2,
            label=f"OLS fit: h* = {fit.params.iloc[1]:.4f}, R² = {fit.rsquared:.4f}")
    ax.plot(x, x, color=MUTED, lw=1, ls="--", label="1:1 line (h = 1)")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Weekly change in CBOT corn futures, dF (cents/bu)")
    ax.set_ylabel("Weekly change in Illinois cash bid, dS (cents/bu)")
    ax.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure2_hedge_regression_train.png", dpi=200)
    plt.close(fig)


def figure3():
    L = pd.read_csv(PROCESSED / "hedge_test_weekly_ledger.csv", parse_dates=["entry_date"])
    N_s = int(pd.read_csv(TABLES / "table2_hedge_ratio.csv")["N_star"].iloc[0])
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.axhline(FULL_HEDGE, color=MUTED, lw=1, ls="--", label=f"1:1 hedge and static min-variance ({N_s} contracts)")
    ax.plot(L["entry_date"], FULL_HEDGE * L["h_rolling"], color=BLUE, lw=1.2, marker="o", ms=3,
            label=f"Rolling min-variance, unrounded (10 x h_t, {ROLL_WINDOW}-pair window)")
    ax.step(L["entry_date"], L["N_rolling"], where="post", color=ORANGE, lw=2, label="Rolling min-variance, contracts held")
    ax.set_ylim(7.5, 10.5)
    ax.set_ylabel("Short contracts (5,000 bu each)")
    ax.set_xlabel("Hedge date (test weeks, 2022-2025)")
    ax.legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure3_rolling_hedge_size_test.png", dpi=200)
    plt.close(fig)


def main(verbose=True):
    FIGURES.mkdir(parents=True, exist_ok=True)
    w = pd.read_csv(PROCESSED / "weekly_clean.csv", parse_dates=["date"])
    p = pd.read_csv(PROCESSED / "eligibility_log.csv", parse_dates=["entry_date", "sale_date"])
    figure1(w)
    figure2(p)
    figure3()
    if verbose:
        print("\nSaved figures:", ", ".join(sorted(f.name for f in FIGURES.glob("*.png"))))


if __name__ == "__main__":
    main()
