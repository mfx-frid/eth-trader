"""
test_paper_buy.py — one-shot plumbing test.

Places a single market BUY on Alpaca paper trading. Intended to be invoked
from the test_paper_buy GitHub Actions workflow, where the secrets live.
If the market is closed (weekend/after-hours), Alpaca queues the order
until the next open, so Saturday invocations still produce a visible
order at app.alpaca.markets.

Env:
  ALPACA_API_KEY, ALPACA_API_SECRET — required
  TEST_TICKER   — default "TSLA"
  TEST_QTY      — default "1"
"""
import os
import sys
import requests

BASE    = "https://paper-api.alpaca.markets"
KEY     = os.environ["ALPACA_API_KEY"]
SECRET  = os.environ["ALPACA_API_SECRET"]
TICKER  = os.environ.get("TEST_TICKER", "TSLA").upper()
QTY     = int(os.environ.get("TEST_QTY", "1"))
HEADERS = {"APCA-API-KEY-ID": KEY, "APCA-API-SECRET-KEY": SECRET}


def account() -> dict:
    r = requests.get(f"{BASE}/v2/account", headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.json()


def place_buy(ticker: str, qty: int) -> dict:
    body = {
        "symbol":        ticker,
        "qty":           str(qty),
        "side":          "buy",
        "type":          "market",
        "time_in_force": "day",
    }
    r = requests.post(f"{BASE}/v2/orders", headers=HEADERS, json=body, timeout=15)
    if r.status_code >= 400:
        print(f"HTTP {r.status_code}: {r.text}")
        r.raise_for_status()
    return r.json()


def main():
    acct = account()
    print(f"Account status: {acct['status']}  buying_power ${acct['buying_power']}")
    print(f"Placing paper BUY: {QTY} × {TICKER}")
    order = place_buy(TICKER, QTY)
    print(f"  id:     {order.get('id')}")
    print(f"  status: {order.get('status')}")
    print(f"  class:  {order.get('order_class')}")
    print(f"  view:   https://app.alpaca.markets/paper/dashboard/overview")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        sys.exit(1)
