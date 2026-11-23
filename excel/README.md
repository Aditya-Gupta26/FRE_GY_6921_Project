# Excel checks

This folder holds the workbooks we use to independently check the proceeds calculation for selected weeks.

The input is `data/processed/hedge_test_weekly_ledger.csv`. It has one row per test week with the entry and
sale prices, the rolling hedge ratio, the number of contracts held, and the sale proceeds for each position.

Sale proceeds equal 50,000 times the sale-date cash bid plus 5,000 times the number of short contracts times
the decline in the futures price (`fut_entry - fut_sale`), with an increase recorded as a loss.
