"""Run the whole baseline in order: cleaning, hedge backtest, forecast, charts.

Every table and chart is saved under results/, and the printed output is also written to
results/baseline_summary.txt.

    python src/run_baseline.py              # uses the saved USDA snapshot in data/raw/
    python src/run_baseline.py --download   # pulls a fresh copy from the USDA API first
"""
import sys

import clean_data
import download_data
import forecast_baseline
import hedge_baseline
import make_figures
import market_snapshot
from config import ROOT


class Tee:
    """Print to the screen and to the summary file at the same time."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def main():
    if "--download" in sys.argv:
        download_data.main()
    out = ROOT / "results" / "baseline_summary.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        stdout = sys.stdout
        sys.stdout = Tee(stdout, f)
        try:
            print("Group 4 working baseline: hedging a 50,000-bushel Illinois corn sale with CBOT corn futures")
            print("Data: USDA AMS Grain Basis, Illinois elevator bid for corn (pull details in data/raw/retrieval_info.json)\n")
            clean_data.main()
            hedge_baseline.main()
            forecast_baseline.main()
            make_figures.main()
            market_snapshot.main()
        finally:
            sys.stdout = stdout
    print(f"\nSummary written to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
