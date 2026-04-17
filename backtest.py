"""
backtest.py — replay the strategy over ~2 weeks of historical daily data.

Pulls daily bars via yfinance for both crypto (ETH-USD, etc.) and stocks.
Runs a rule-based decision (RSI + MACD + SMA) that mirrors the live traders'
execution flow: MIN_CONFIDENCE gate, MAX_POSITION sizing, 3-tranche ladder,
TRAIL_PERCENT trailing stop. No Anthropic API calls — fast and deterministic.

Usage:
  py backtest.py                     # 14-day window, all tickers
  py backtest.py --days 21           # 21-day window
  py backtest.py --ticker NVDA       # single ticker
  py backtest.py --type crypto       # crypto only
"""

import argparse
import yfinance as yf

# ── Strategy parameters (mirror live scripts) ────────────────────────────────
START_CASH       = 10000.0
MAX_POSITION     = 0.15
MIN_CONFIDENCE   = 7
TRAIL_PERCENT    = 3.0
LADDER_STEPS_PCT = [0.0, 2.0, 4.0]
WARMUP_DAYS      = 40  # need >=35 for MACD signal; 40 for margin

CRYPTO_TICKERS = ["ETH-USD", "BTC-USD", "SOL-USD", "XRP-USD", "LTC-USD", "DOGE-USD"]
STOCK_TICKERS  = ["NVDA", "AAPL", "TSLA", "META", "AMD", "COIN"]


# ── Indicators ────────────────────────────────────────────────────────────────
def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains  = [d for d in deltas[-period:] if d > 0]
    losses = [-d for d in deltas[-period:] if d < 0]
    avg_g  = sum(gains)  / period if gains  else 0.0
    avg_l  = sum(losses) / period if losses else 0.001
    return 100 - (100 / (1 + avg_g / avg_l))


def calc_macd(closes, fast=12, slow=26, signal=9):
    if len(closes) < slow + signal - 1:
        return None, None, None
    def _ema(values, period):
        a = 2 / (period + 1)
        out = [sum(values[:period]) / period]
        for v in values[period:]:
            out.append(v * a + out[-1] * (1 - a))
        return out
    ef = _ema(closes, fast)
    es = _ema(closes, slow)
    ef_a = ef[slow - fast:]
    macd_line = [f - s for f, s in zip(ef_a, es)]
    sig_line  = _ema(macd_line, signal)
    macd_a    = macd_line[signal - 1:]
    hist_line = [m - s for m, s in zip(macd_a, sig_line)]
    return macd_a[-1], sig_line[-1], hist_line[-1]


def calc_sma(closes, period):
    if len(closes) < period:
        return closes[-1]
    return sum(closes[-period:]) / period


# ── Strategy: rule-based mirror of Claude's usual behaviour ──────────────────
def decide(closes, price, avg_buy, holding, pending):
    """Return (action, confidence). Simple rules; tune as needed."""
    rsi = calc_rsi(closes)
    _, _, hist = calc_macd(closes)
    if hist is None:
        return ("HOLD", 0)
    sma14 = calc_sma(closes, 14)

    if holding:
        if rsi > 70:
            return ("SELL", 8)
        if hist < 0 and price > avg_buy * 1.03:
            return ("SELL", 7)
        return ("HOLD", 5)

    if pending:
        return ("HOLD", 5)  # ladder already placed; wait for fills
    if rsi < 35 and hist > 0:
        return ("BUY", 8)
    if rsi < 45 and hist > 0 and price < sma14:
        return ("BUY", 7)
    return ("HOLD", 5)


# ── Simulation core ───────────────────────────────────────────────────────────
def new_portfolio():
    return {
        "cash":     START_CASH,
        "position": None,        # {"qty", "avg_buy", "peak"}
        "pending":  [],          # [{"qty", "limit_price"}]
        "pnl":      0.0,
        "trades":   [],          # (date, kind, qty, price, pnl)
    }


def step_bar(p, date, price):
    """Advance one bar: fill pending, check trailing stop. Mutates p."""
    # Pending ladder fills
    still = []
    for o in p["pending"]:
        if price <= o["limit_price"] and p["cash"] >= o["qty"] * o["limit_price"]:
            cost = o["qty"] * o["limit_price"]
            pos  = p["position"] or {"qty": 0, "avg_buy": 0, "peak": price}
            new_q  = pos["qty"] + o["qty"]
            new_a  = (pos["avg_buy"] * pos["qty"] + o["limit_price"] * o["qty"]) / new_q
            p["position"] = {"qty": new_q, "avg_buy": new_a,
                             "peak": max(pos["peak"], price)}
            p["cash"] -= cost
            p["trades"].append((date, "LIMIT_FILL", o["qty"], o["limit_price"], 0.0))
        else:
            still.append(o)
    p["pending"] = still

    # Trailing-stop check
    pos = p["position"]
    if pos:
        pos["peak"] = max(pos["peak"], price)
        stop = pos["peak"] * (1 - TRAIL_PERCENT / 100)
        if price <= stop:
            proceeds = pos["qty"] * price
            pnl      = proceeds - pos["avg_buy"] * pos["qty"]
            p["cash"]    += proceeds
            p["pnl"]     += pnl
            p["trades"].append((date, "TRAIL_STOP", pos["qty"], price, pnl))
            p["position"] = None
            p["pending"]  = []


def apply_signal(p, date, price, action, confidence):
    pos = p["position"]
    holding = pos is not None and pos["qty"] > 0

    if action == "BUY" and not holding and confidence >= MIN_CONFIDENCE and not p["pending"]:
        budget  = p["cash"] * MAX_POSITION * (confidence / 10)
        tranche = budget / len(LADDER_STEPS_PCT)

        # Tranche 1: market
        t1_qty = tranche / price
        p["position"] = {"qty": t1_qty, "avg_buy": price, "peak": price}
        p["cash"]    -= t1_qty * price
        p["trades"].append((date, "BUY", t1_qty, price, 0.0))

        # Tranches 2+: pending
        for step in LADDER_STEPS_PCT[1:]:
            lp = price * (1 - step / 100)
            p["pending"].append({"qty": tranche / lp, "limit_price": lp})

    elif action == "SELL" and holding:
        proceeds = pos["qty"] * price
        pnl      = proceeds - pos["avg_buy"] * pos["qty"]
        p["cash"]    += proceeds
        p["pnl"]     += pnl
        p["trades"].append((date, "SELL", pos["qty"], price, pnl))
        p["position"] = None
        p["pending"]  = []


def run_backtest(ticker, days):
    total = WARMUP_DAYS + days + 5
    hist  = yf.Ticker(ticker).history(period=f"{total}d")
    if hist.empty or len(hist) < WARMUP_DAYS + 2:
        return None

    closes = hist["Close"].tolist()
    dates  = [d.strftime("%Y-%m-%d") for d in hist.index]

    p     = new_portfolio()
    start = max(WARMUP_DAYS, len(closes) - days)

    for i in range(start, len(closes)):
        window = closes[:i + 1]
        price  = closes[i]
        date   = dates[i]

        step_bar(p, date, price)

        pos        = p["position"]
        avg_buy    = pos["avg_buy"] if pos else 0
        holding    = pos is not None
        has_pendg  = len(p["pending"]) > 0

        action, conf = decide(window, price, avg_buy, holding, has_pendg)
        apply_signal(p, date, price, action, conf)

    # Close remaining position at final bar
    final = closes[-1]
    if p["position"]:
        pos      = p["position"]
        proceeds = pos["qty"] * final
        pnl      = proceeds - pos["avg_buy"] * pos["qty"]
        p["cash"]    += proceeds
        p["pnl"]     += pnl
        p["trades"].append((dates[-1], "CLOSE", pos["qty"], final, pnl))
        p["position"] = None

    p["first_date"] = dates[start]
    p["last_date"]  = dates[-1]
    p["n_bars"]     = len(closes) - start
    return p


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14, help="Backtest window (days)")
    ap.add_argument("--ticker", default=None, help="Single ticker (e.g. NVDA or ETH-USD)")
    ap.add_argument("--type", choices=["crypto", "stock", "all"], default="all")
    args = ap.parse_args()

    if args.ticker:
        tickers = [args.ticker]
    else:
        tickers = []
        if args.type in ("all", "crypto"): tickers += CRYPTO_TICKERS
        if args.type in ("all", "stock"):  tickers += STOCK_TICKERS

    print(f"\n{'='*66}")
    print(f"  Strategy backtest  •  {args.days}-day window  •  warmup {WARMUP_DAYS}d")
    print(f"  start cash: ${START_CASH:,.0f}  ·  MIN_CONF: {MIN_CONFIDENCE}  ·  "
          f"trail: {TRAIL_PERCENT}%  ·  ladder: {LADDER_STEPS_PCT}")
    print(f"{'='*66}\n")

    header = f"  {'Ticker':<10} {'Window':<24} {'Trades':>7} {'P&L':>10} {'Final':>10} {'ROI':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    total_pnl = 0.0
    total_n   = 0
    for t in tickers:
        r = run_backtest(t, args.days)
        if r is None:
            print(f"  {t:<10} (no data)")
            continue
        roi = (r["cash"] - START_CASH) / START_CASH * 100
        print(f"  {t:<10} {r['first_date']}→{r['last_date']}  "
              f"{len(r['trades']):>5}  ${r['pnl']:+9.2f}  ${r['cash']:>9.2f}  {roi:+6.2f}%")
        total_pnl += r["pnl"]
        total_n   += len(r["trades"])

    print("  " + "-" * (len(header) - 2))
    print(f"  Aggregate: {total_n} trades, total P&L ${total_pnl:+.2f}\n")


if __name__ == "__main__":
    main()
