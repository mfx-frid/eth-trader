"""
Alpaca Paper Trading — AI Stock Advisor Script
===============================================
What this does:
  1. Connects to Alpaca Paper Trading API
  2. Fetches live price + technical indicators for multiple stocks
  3. Pulls recent news via Yahoo Finance
  4. Sends everything to Claude API for analysis
  5. Places real paper orders on Alpaca (visible in dashboard)
  6. Logs every decision to alpaca_trade_log.csv

Requirements:
  pip install requests anthropic yfinance

Configuration:
  Fill in the three variables in the CONFIGURATION section below.
"""

import os
import csv
import json
import datetime
import requests
import yfinance as yf
import anthropic

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

ALPACA_API_KEY    = os.environ.get("ALPACA_API_KEY",    "")
ALPACA_API_SECRET = os.environ.get("ALPACA_API_SECRET", "")
ANTHROPIC_API_KEY   = os.environ.get("ANTHROPIC_API_KEY",   "")

# Stocks to analyse each run — add or remove as you like
TICKERS = ["NVDA", "AAPL", "TSLA", "META", "AMD", "COIN"]

PAPER_BASE_URL  = "https://paper-api.alpaca.markets"
DATA_BASE_URL   = "https://data.alpaca.markets"
LOG_FILE        = "alpaca_trade_log.csv"

# Risk controls — adjust these to your comfort level
MAX_POSITION_USD = 2000.0    # Max USD per single trade
MAX_OPEN_TRADES  = 5        # Max simultaneous open positions

# ── ALPACA API ────────────────────────────────────────────────────────────────

def _alpaca_headers() -> dict:
    """Simple key/secret headers — much easier than Coinbase JWT."""
    return {
        "APCA-API-KEY-ID":     ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
        "Content-Type":        "application/json",
    }


def get_account() -> dict:
    """Fetch paper account details — balance, buying power etc."""
    resp = requests.get(
        f"{PAPER_BASE_URL}/v2/account",
        headers=_alpaca_headers(),
        timeout=10
    )
    resp.raise_for_status()
    return resp.json()


def get_positions() -> list[dict]:
    """Get all currently open paper positions."""
    resp = requests.get(
        f"{PAPER_BASE_URL}/v2/positions",
        headers=_alpaca_headers(),
        timeout=10
    )
    resp.raise_for_status()
    return resp.json()


def get_open_orders() -> list[dict]:
    """Get all currently open orders."""
    resp = requests.get(
        f"{PAPER_BASE_URL}/v2/orders?status=open",
        headers=_alpaca_headers(),
        timeout=10
    )
    resp.raise_for_status()
    return resp.json()


def place_order(symbol: str, qty: float, side: str,
                stop_loss: float = None) -> dict:
    """Place a paper market order on Alpaca."""
    order = {
        "symbol":        symbol,
        "qty":           str(qty),
        "side":          side,          # "buy" or "sell"
        "type":          "market",
        "time_in_force": "day",
    }
    # Attach stop loss as bracket order if provided
    if stop_loss and side == "buy":
        order["order_class"]  = "bracket"
        order["stop_loss"]    = {"stop_price": str(round(stop_loss, 2))}

    resp = requests.post(
        f"{PAPER_BASE_URL}/v2/orders",
        headers=_alpaca_headers(),
        json=order,
        timeout=10
    )
    resp.raise_for_status()
    return resp.json()


def get_latest_price(symbol: str) -> float:
    """Fetch latest trade price from Alpaca data API."""
    resp = requests.get(
        f"{DATA_BASE_URL}/v2/stocks/{symbol}/trades/latest",
        headers=_alpaca_headers(),
        timeout=10
    )
    resp.raise_for_status()
    data = resp.json()
    return float(data.get("trade", {}).get("p", 0))


# ── MARKET DATA & INDICATORS ──────────────────────────────────────────────────

def fetch_market_data(ticker: str, days: int = 14) -> dict:
    """Pull OHLCV + calculate indicators via Yahoo Finance."""
    stock = yf.Ticker(ticker)
    hist  = stock.history(period=f"{days}d")

    if hist.empty:
        raise ValueError(f"No data for {ticker}")

    closes  = hist["Close"].tolist()
    volumes = hist["Volume"].tolist()
    dates   = [d.strftime("%Y-%m-%d") for d in hist.index]

    sma7  = round(sum(closes[-7:])  / min(7,  len(closes)), 2)
    sma14 = round(sum(closes[-14:]) / min(14, len(closes)), 2)

    deltas   = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains    = [d for d in deltas if d > 0]
    losses   = [-d for d in deltas if d < 0]
    avg_gain = sum(gains[-14:])  / 14 if gains  else 0
    avg_loss = sum(losses[-14:]) / 14 if losses else 0.001
    rsi      = round(100 - (100 / (1 + avg_gain / avg_loss)), 1)

    price_now = closes[-1]
    price_1w  = closes[-7] if len(closes) >= 7 else closes[0]
    change_1w = round(((price_now - price_1w) / price_1w) * 100, 2)

    return {
        "ticker":        ticker,
        "current_price": round(price_now, 2),
        "change_7d_pct": change_1w,
        "sma_7":         sma7,
        "sma_14":        sma14,
        "rsi_14":        rsi,
        "avg_volume_7d": round(sum(volumes[-7:]) / 7),
        "price_history": list(zip(dates, [round(c, 2) for c in closes])),
    }


def fetch_news(ticker: str, max_items: int = 5) -> list[str]:
    """Pull recent headlines via Yahoo Finance."""
    try:
        stock     = yf.Ticker(ticker)
        news      = stock.news or []
        headlines = []
        for item in news[:max_items]:
            content = item.get("content", {})
            title   = content.get("title") or item.get("title", "")
            if title:
                headlines.append(title)
        return headlines if headlines else ["No recent news found."]
    except Exception:
        return ["Could not fetch news."]


# ── CLAUDE ANALYSIS ───────────────────────────────────────────────────────────

def build_prompt(market: dict, headlines: list[str],
                 account: dict, positions: list[dict]) -> str:
    price_table = "\n".join(
        f"  {date}: ${price}" for date, price in market["price_history"]
    )
    news_block = "\n".join(f"  - {h}" for h in headlines)

    # Current position in this stock if any
    held = next((p for p in positions
                 if p["symbol"] == market["ticker"]), None)
    position_info = (
        f"Currently holding {held['qty']} shares "
        f"(avg entry: ${float(held['avg_entry_price']):.2f}, "
        f"unrealised P&L: ${float(held['unrealized_pl']):.2f})"
        if held else "No current position in this stock."
    )

    return f"""You are a cautious quantitative trading analyst specialising in US stocks.
Analyse the following data and provide a clear trading recommendation.

## Stock: {market['ticker']}

### Price data (last {len(market['price_history'])} trading days)
{price_table}

### Key indicators
- Current price:  ${market['current_price']}
- 7-day change:   {market['change_7d_pct']:+.1f}%
- SMA-7:          ${market['sma_7']}
- SMA-14:         ${market['sma_14']}
- RSI-14:         {market['rsi_14']} (oversold <30, overbought >70)
- Avg daily vol:  {market['avg_volume_7d']:,}

### Current position
{position_info}

### Paper account
- Buying power:   ${float(account.get('buying_power', 0)):,.2f}
- Portfolio value: ${float(account.get('portfolio_value', 0)):,.2f}

### Recent news
{news_block}

---

Respond with a JSON object in EXACTLY this format (no markdown fences):
{{
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": 1-10,
  "qty": <number of shares or null>,
  "price_target": <number or null>,
  "stop_loss": <number or null>,
  "reasoning": "<2-3 sentence explanation>",
  "key_risks": ["<risk 1>", "<risk 2>"],
  "time_horizon": "short-term (days)" | "medium-term (weeks)" | "long-term (months)"
}}

Rules:
- Only recommend BUY if confidence >= 7
- Only recommend SELL if we currently hold the stock
- When in doubt, HOLD
- Keep qty reasonable for a ${MAX_POSITION_USD} max position size
- This is paper trading — be educational but realistic"""


def call_claude(prompt: str) -> dict:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    msg    = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = msg.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)


# ── ORDER EXECUTION ───────────────────────────────────────────────────────────

def execute_trade(rec: dict, market: dict,
                  positions: list[dict], open_orders: list[dict]) -> dict:
    """Execute a real paper trade on Alpaca based on Claude's recommendation."""
    action  = rec["action"]
    ticker  = market["ticker"]
    price   = market["current_price"]
    result  = {"action": action, "ticker": ticker,
                "shares": 0, "usd_value": 0, "note": ""}

    if action == "BUY":
        # Safety checks
        open_count = len(positions)
        if open_count >= MAX_OPEN_TRADES:
            result["note"] = f"Max {MAX_OPEN_TRADES} positions reached — skipped"
            return result
        if rec["confidence"] < 7:
            result["note"] = f"Confidence {rec['confidence']}/10 too low — skipped"
            return result
        # Already holding this stock?
        already_held = any(p["symbol"] == ticker for p in positions)
        if already_held:
            result["note"] = "Already holding — skipped"
            return result

        # Calculate share quantity
        usd_amount = min(MAX_POSITION_USD,
                        MAX_POSITION_USD * (rec["confidence"] / 10))
        qty        = max(1, int(usd_amount / price))
        stop_loss  = rec.get("stop_loss")

        try:
            order = place_order(ticker, qty, "buy", stop_loss)
            result.update({
                "shares":    qty,
                "usd_value": round(qty * price, 2),
                "order_id":  order.get("id", ""),
                "note":      "Order placed ✅"
            })
        except requests.HTTPError as e:
            result["note"] = f"Order failed: {e}"

    elif action == "SELL":
        held = next((p for p in positions if p["symbol"] == ticker), None)
        if not held:
            result["note"] = "No position to sell — skipped"
            return result
        qty = int(float(held["qty"]))
        try:
            order = place_order(ticker, qty, "sell")
            result.update({
                "shares":    qty,
                "usd_value": round(qty * price, 2),
                "order_id":  order.get("id", ""),
                "note":      "Sell order placed ✅"
            })
        except requests.HTTPError as e:
            result["note"] = f"Order failed: {e}"

    else:
        result["note"] = "Holding — no order placed"

    return result


# ── CSV LOGGING ───────────────────────────────────────────────────────────────

def log_to_csv(market: dict, rec: dict, trade: dict, account: dict) -> None:
    file_exists = os.path.isfile(LOG_FILE)
    fieldnames  = [
        "timestamp", "ticker", "price", "change_7d_pct",
        "sma_7", "sma_14", "rsi_14",
        "action", "confidence", "time_horizon",
        "price_target", "stop_loss", "reasoning",
        "trade_shares", "trade_usd", "trade_note",
        "portfolio_value", "buying_power",
    ]
    row = {
        "timestamp":       datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "ticker":          market["ticker"],
        "price":           market["current_price"],
        "change_7d_pct":   market["change_7d_pct"],
        "sma_7":           market["sma_7"],
        "sma_14":          market["sma_14"],
        "rsi_14":          market["rsi_14"],
        "action":          rec["action"],
        "confidence":      rec["confidence"],
        "time_horizon":    rec.get("time_horizon", ""),
        "price_target":    rec.get("price_target", ""),
        "stop_loss":       rec.get("stop_loss", ""),
        "reasoning":       rec.get("reasoning", ""),
        "trade_shares":    trade.get("shares", 0),
        "trade_usd":       trade.get("usd_value", 0),
        "trade_note":      trade.get("note", ""),
        "portfolio_value": round(float(account.get("portfolio_value", 0)), 2),
        "buying_power":    round(float(account.get("buying_power", 0)), 2),
    }
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ── REPORT ────────────────────────────────────────────────────────────────────

def print_report(market: dict, rec: dict,
                 trade: dict, account: dict) -> None:
    sep = "─" * 62
    print(f"\n{sep}")
    action = rec["action"]
    icon   = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(action, "⚪")
    print(f"  {market['ticker']}  •  ${market['current_price']}  "
          f"({market['change_7d_pct']:+.1f}% / 7d)  "
          f"|  RSI: {market['rsi_14']}")
    print(f"  {icon}  {action}  (Confidence: {rec['confidence']}/10)  "
          f"—  {rec.get('time_horizon', '')}")
    print(f"  {rec['reasoning']}")
    if trade.get("note"):
        print(f"  📋  {trade['note']}")
    if trade.get("shares"):
        side = "Buy" if action == "BUY" else "Sell"
        print(f"      {side} {trade['shares']} shares "
              f"@ ${market['current_price']} = ${trade['usd_value']}")
    risks = rec.get("key_risks", [])
    for r in risks:
        print(f"  ⚠  {r}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    if "YOUR_ALPACA" in ALPACA_API_KEY or "YOUR_ANTHROPIC" in ANTHROPIC_API_KEY:
        print("ERROR: Fill in your API keys in the CONFIGURATION section.")
        return

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n{'═'*62}")
    print(f"  Claude AI Stock Advisor — Alpaca Paper  •  {now}")
    print(f"{'═'*62}")

    # Fetch account info once
    try:
        account     = get_account()
        positions   = get_positions()
        open_orders = get_open_orders()
    except requests.HTTPError as e:
        print(f"Alpaca API error: {e}")
        print("Check your API key and secret.")
        return

    port_val = float(account.get("portfolio_value", 0))
    buying   = float(account.get("buying_power", 0))
    print(f"\n  💼  Portfolio: ${port_val:,.2f}  "
          f"|  Buying power: ${buying:,.2f}  "
          f"|  Open positions: {len(positions)}")

    # Analyse each ticker
    for ticker in TICKERS:
        print(f"\n  Fetching {ticker}...")
        try:
            market    = fetch_market_data(ticker)
            headlines = fetch_news(ticker)
            prompt    = build_prompt(market, headlines, account, positions)

            print(f"  Calling Claude for {ticker}...")
            rec   = call_claude(prompt)
            trade = execute_trade(rec, market, positions, open_orders)

            print_report(market, rec, trade, account)
            log_to_csv(market, rec, trade, account)

            # Refresh positions after a trade
            if trade.get("shares", 0) > 0:
                positions = get_positions()

        except Exception as e:
            print(f"  Error processing {ticker}: {e}")
            continue

    print(f"\n{'═'*62}")
    print(f"  ⚠  Paper trading only. Not financial advice.")
    print(f"  📁  Logged to {LOG_FILE}")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    main()
