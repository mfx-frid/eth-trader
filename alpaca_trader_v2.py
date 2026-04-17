"""
Alpaca Paper Trading — AI Stock Advisor v2
==========================================
Fixes:
  - Removed bracket/stop-loss orders (caused 422 errors)
  - Market hours check — skips order placement if market closed
  - Weekday-aware (won't waste API calls on weekends via schedule)
  - Increased MAX_POSITION_USD for more realistic trades
  - Portfolio value tracked properly in CSV

Requirements:
  pip install requests anthropic yfinance
"""

import os
import csv
import json
import time
import datetime
import requests
import yfinance as yf
import anthropic

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

ALPACA_API_KEY    = os.environ.get("ALPACA_API_KEY",    "")
ALPACA_API_SECRET = os.environ.get("ALPACA_API_SECRET", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

TICKERS = ["NVDA", "AAPL", "TSLA", "META", "AMD", "COIN"]

PAPER_BASE_URL   = "https://paper-api.alpaca.markets"
DATA_BASE_URL    = "https://data.alpaca.markets"
LOG_FILE         = "alpaca_trade_log.csv"

MAX_POSITION_USD = 2000.0   # Max USD total across all 3 ladder tranches
MAX_OPEN_TRADES  = 4        # Max simultaneous positions
MIN_CONFIDENCE   = 7        # Minimum Claude confidence to BUY
TRAIL_PERCENT    = 3.0      # Trailing stop distance as % below peak
TRAIL_FILL_WAIT  = 2        # Seconds to wait for BUY fill before placing stop
LADDER_STEPS_PCT = [0.0, 2.0, 4.0]  # Tranche entry discounts: market, -2%, -4%

# ── ALPACA API ────────────────────────────────────────────────────────────────

def _headers() -> dict:
    return {
        "APCA-API-KEY-ID":     ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
        "Content-Type":        "application/json",
    }


def get_account() -> dict:
    resp = requests.get(f"{PAPER_BASE_URL}/v2/account",
                        headers=_headers(), timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_positions() -> list[dict]:
    resp = requests.get(f"{PAPER_BASE_URL}/v2/positions",
                        headers=_headers(), timeout=10)
    resp.raise_for_status()
    return resp.json()


def is_market_open() -> bool:
    """Check if US market is currently open via Alpaca clock endpoint."""
    try:
        resp = requests.get(f"{PAPER_BASE_URL}/v2/clock",
                            headers=_headers(), timeout=10)
        resp.raise_for_status()
        return resp.json().get("is_open", False)
    except Exception:
        return False


def place_order(symbol: str, qty: int, side: str) -> dict:
    """Place a simple market order — no bracket, no stop loss."""
    order = {
        "symbol":        symbol,
        "qty":           str(qty),
        "side":          side,
        "type":          "market",
        "time_in_force": "day",
    }
    resp = requests.post(f"{PAPER_BASE_URL}/v2/orders",
                         headers=_headers(), json=order, timeout=10)
    resp.raise_for_status()
    return resp.json()


def place_trailing_stop(symbol: str, qty: int,
                        trail_percent: float = TRAIL_PERCENT) -> dict:
    """Place a GTC trailing-stop sell that follows price by `trail_percent` %."""
    order = {
        "symbol":        symbol,
        "qty":           str(qty),
        "side":          "sell",
        "type":          "trailing_stop",
        "trail_percent": str(trail_percent),
        "time_in_force": "gtc",
    }
    resp = requests.post(f"{PAPER_BASE_URL}/v2/orders",
                         headers=_headers(), json=order, timeout=10)
    resp.raise_for_status()
    return resp.json()


def place_limit_buy(symbol: str, qty: int, limit_price: float) -> dict:
    """Place a GTC limit buy — used for lower ladder rungs."""
    order = {
        "symbol":        symbol,
        "qty":           str(qty),
        "side":          "buy",
        "type":          "limit",
        "limit_price":   f"{limit_price:.2f}",
        "time_in_force": "gtc",
    }
    resp = requests.post(f"{PAPER_BASE_URL}/v2/orders",
                         headers=_headers(), json=order, timeout=10)
    resp.raise_for_status()
    return resp.json()


def cancel_open_orders_for(symbol: str) -> int:
    """Cancel any open orders for `symbol` (e.g. stale trailing stops) so a
    manual SELL doesn't collide with them. Returns count cancelled."""
    resp = requests.get(
        f"{PAPER_BASE_URL}/v2/orders",
        headers=_headers(),
        params={"status": "open", "symbols": symbol},
        timeout=10,
    )
    resp.raise_for_status()
    cancelled = 0
    for o in resp.json():
        try:
            requests.delete(
                f"{PAPER_BASE_URL}/v2/orders/{o['id']}",
                headers=_headers(), timeout=10,
            ).raise_for_status()
            cancelled += 1
        except requests.HTTPError:
            pass
    return cancelled


# ── MARKET DATA ───────────────────────────────────────────────────────────────

def calculate_vwap(closes: list[float], volumes: list[float],
                   period: int = 14) -> float | None:
    """Volume-weighted average price over last `period` bars."""
    if len(closes) < period or len(volumes) < period:
        return None
    c  = closes[-period:]
    v  = volumes[-period:]
    tv = sum(v)
    if tv <= 0:
        return None
    return round(sum(ci * vi for ci, vi in zip(c, v)) / tv, 4)


def calculate_macd(closes: list[float],
                   fast: int = 12, slow: int = 26,
                   signal: int = 9) -> dict:
    """Return latest MACD line, signal, histogram. None if insufficient data."""
    if len(closes) < slow + signal - 1:
        return {"macd": None, "signal": None, "hist": None}

    def _ema(values: list[float], period: int) -> list[float]:
        alpha = 2 / (period + 1)
        seed  = sum(values[:period]) / period
        out   = [seed]
        for v in values[period:]:
            out.append(v * alpha + out[-1] * (1 - alpha))
        return out

    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    ema_fast_aligned = ema_fast[slow - fast:]
    macd_line = [f - s for f, s in zip(ema_fast_aligned, ema_slow)]
    sig_line  = _ema(macd_line, signal)
    macd_aligned = macd_line[signal - 1:]
    hist_line = [m - s for m, s in zip(macd_aligned, sig_line)]
    return {
        "macd":   round(macd_aligned[-1], 4),
        "signal": round(sig_line[-1], 4),
        "hist":   round(hist_line[-1], 4),
    }


def fetch_market_data(ticker: str, days: int = 60) -> dict:
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

    macd    = calculate_macd(closes)
    vwap_14 = calculate_vwap(closes, volumes, 14)

    price_now = closes[-1]
    price_1w  = closes[-7] if len(closes) >= 7 else closes[0]
    change_1w = round(((price_now - price_1w) / price_1w) * 100, 2)

    # Keep only last 14 trading days in history tail for prompt + dashboard
    hist_tail = list(zip(dates[-14:], [round(c, 2) for c in closes[-14:]]))

    return {
        "ticker":        ticker,
        "current_price": round(price_now, 2),
        "change_7d_pct": change_1w,
        "sma_7":         sma7,
        "sma_14":        sma14,
        "rsi_14":        rsi,
        "macd":          macd["macd"],
        "macd_signal":   macd["signal"],
        "macd_hist":     macd["hist"],
        "vwap_14":       vwap_14,
        "avg_volume_7d": round(sum(volumes[-7:]) / 7),
        "price_history": hist_tail,
    }


def fetch_news(ticker: str, max_items: int = 5) -> list[str]:
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


def fetch_capitol_trades(ticker: str, max_items: int = 5) -> list[str]:
    """Fetch recent US politician disclosures for a ticker from capitoltrades.com.
    Uses unofficial API — wrapped in try/except, returns [] on any failure."""
    try:
        resp = requests.get(
            "https://bff.capitoltrades.com/trades",
            params={
                "assetTickers": ticker,
                "pageSize":     max_items,
                "txType":       "buy,sell",
            },
            headers={"User-Agent": "Mozilla/5.0 (eth-trader)"},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
    except Exception:
        return []

    lines = []
    for t in data[:max_items]:
        try:
            pol   = t.get("politician", {}) or {}
            name  = f"{pol.get('firstName','')} {pol.get('lastName','')}".strip() or "Unknown"
            party = pol.get("party", "") or "?"
            chamb = pol.get("chamber", "") or ""
            side  = (t.get("txType", "") or "").upper()
            size  = t.get("size", "") or t.get("value", "") or "size undisclosed"
            date  = t.get("txDate", "") or t.get("pubDate", "")[:10]
            lines.append(f"{date} — {name} ({party} {chamb}) {side} {size}")
        except Exception:
            continue
    return lines


# ── CLAUDE ANALYSIS ───────────────────────────────────────────────────────────

def build_prompt(market: dict, headlines: list[str],
                 account: dict, positions: list[dict],
                 politicians: list[str] = None) -> str:
    price_table = "\n".join(
        f"  {date}: ${price}" for date, price in market["price_history"]
    )
    news_block    = "\n".join(f"  - {h}" for h in headlines)
    politicians   = politicians or []
    pol_block     = (
        "\n".join(f"  - {p}" for p in politicians)
        if politicians else "  - No recent politician disclosures found."
    )
    held = next((p for p in positions
                 if p["symbol"] == market["ticker"]), None)
    position_info = (
        f"Holding {held['qty']} shares "
        f"(avg entry: ${float(held['avg_entry_price']):.2f}, "
        f"unrealised P&L: ${float(held['unrealized_pl']):.2f})"
        if held else "No current position."
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
- MACD-12/26:     {market['macd']}  signal: {market['macd_signal']}  hist: {market['macd_hist']} (hist>0 bullish momentum, hist<0 bearish; sign flip = crossover)
- VWAP-14:        ${market['vwap_14']} (price above VWAP = paying premium, below = discount)
- Avg daily vol:  {market['avg_volume_7d']:,}

### Current position
{position_info}

### Paper account
- Portfolio value: ${float(account.get('portfolio_value', 0)):,.2f}
- Buying power:    ${float(account.get('buying_power', 0)):,.2f}

### Recent news
{news_block}

### Recent US politician disclosures (capitoltrades.com)
{pol_block}

---

Respond with JSON ONLY — no markdown fences:
{{
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": 1-10,
  "price_target": <number or null>,
  "stop_loss": <number or null>,
  "reasoning": "<2-3 sentences>",
  "key_risks": ["<risk 1>", "<risk 2>"],
  "time_horizon": "short-term (days)" | "medium-term (weeks)" | "long-term (months)"
}}

Rules:
- BUY only if confidence >= {MIN_CONFIDENCE}
- SELL only if currently holding the stock
- When in doubt HOLD
- Max position size is ${MAX_POSITION_USD}"""


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
                  positions: list[dict],
                  market_open: bool) -> dict:
    action = rec["action"]
    ticker = market["ticker"]
    price  = market["current_price"]
    result = {"action": action, "ticker": ticker,
               "shares": 0, "usd_value": 0, "note": ""}

    if action == "BUY":
        if rec["confidence"] < MIN_CONFIDENCE:
            result["note"] = f"Confidence {rec['confidence']}/10 too low — skipped"
            return result
        if len(positions) >= MAX_OPEN_TRADES:
            result["note"] = f"Max {MAX_OPEN_TRADES} positions reached — skipped"
            return result
        if any(p["symbol"] == ticker for p in positions):
            result["note"] = "Already holding — skipped"
            return result
        if not market_open:
            result["note"] = "Market closed — signal logged, no order placed"
            return result

        # Ladder: split budget across tranches. Tranche 1 = market (immediate),
        # rest = limit orders at successive discounts, GTC.
        n_tranches  = len(LADDER_STEPS_PCT)
        tranche_usd = MAX_POSITION_USD / n_tranches
        t1_qty      = max(1, int(tranche_usd / price))
        total_qty   = t1_qty  # tracks filled shares (tranche 1 only, initially)
        note_parts  = []
        try:
            mkt_order = place_order(ticker, t1_qty, "buy")
            note_parts.append(f"T1 mkt {t1_qty}sh ✅ ({mkt_order.get('id','')[:8]})")

            # Attach trailing stop to the just-filled market tranche.
            time.sleep(TRAIL_FILL_WAIT)
            try:
                ts = place_trailing_stop(ticker, t1_qty, TRAIL_PERCENT)
                note_parts.append(f"trail {TRAIL_PERCENT}% ({ts.get('id','')[:8]})")
            except requests.HTTPError as te:
                note_parts.append(f"trail failed: {te}")

            # Lower rungs — limit buys at -2%, -4% etc.
            for step in LADDER_STEPS_PCT[1:]:
                limit_px = round(price * (1 - step / 100), 2)
                rung_qty = max(1, int(tranche_usd / limit_px))
                try:
                    lo = place_limit_buy(ticker, rung_qty, limit_px)
                    note_parts.append(
                        f"T-{step:g}% lim {rung_qty}sh @${limit_px} ({lo.get('id','')[:8]})"
                    )
                except requests.HTTPError as le:
                    note_parts.append(f"T-{step:g}% limit failed: {le}")

            result.update({
                "shares":    total_qty,
                "usd_value": round(total_qty * price, 2),
                "note":      " | ".join(note_parts),
            })
        except requests.HTTPError as e:
            result["note"] = f"Ladder failed at tranche 1: {e}"

    elif action == "SELL":
        held = next((p for p in positions if p["symbol"] == ticker), None)
        if not held:
            result["note"] = "No position to sell — skipped"
            return result
        if not market_open:
            result["note"] = "Market closed — sell signal logged, no order placed"
            return result
        qty = int(float(held["qty"]))
        try:
            n_cancelled = cancel_open_orders_for(ticker)
            order = place_order(ticker, qty, "sell")
            note  = f"Sell order placed ✅ (id: {order.get('id','')[:8]})"
            if n_cancelled:
                note += f" (cancelled {n_cancelled} stale order{'s' if n_cancelled>1 else ''})"
            result.update({
                "shares":    qty,
                "usd_value": round(qty * price, 2),
                "note":      note,
            })
        except requests.HTTPError as e:
            result["note"] = f"Order failed: {e}"
    else:
        result["note"] = "Holding — no order"

    return result


# ── CSV LOGGING ───────────────────────────────────────────────────────────────

def log_to_csv(market: dict, rec: dict,
               trade: dict, account: dict) -> None:
    file_exists = os.path.isfile(LOG_FILE)
    fieldnames  = [
        "timestamp", "ticker", "price", "change_7d_pct",
        "sma_7", "sma_14", "rsi_14", "market_open",
        "action", "confidence", "time_horizon",
        "price_target", "stop_loss", "reasoning",
        "trade_shares", "trade_usd", "trade_note",
        "portfolio_value", "buying_power",
        "macd", "macd_signal", "macd_hist", "vwap_14",
    ]
    row = {
        "timestamp":       datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "ticker":          market["ticker"],
        "price":           market["current_price"],
        "change_7d_pct":   market["change_7d_pct"],
        "sma_7":           market["sma_7"],
        "sma_14":          market["sma_14"],
        "rsi_14":          market["rsi_14"],
        "market_open":     trade.get("market_open", ""),
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
        "macd":            market["macd"]        if market["macd"]        is not None else "",
        "macd_signal":     market["macd_signal"] if market["macd_signal"] is not None else "",
        "macd_hist":       market["macd_hist"]   if market["macd_hist"]   is not None else "",
        "vwap_14":         market["vwap_14"]     if market["vwap_14"]     is not None else "",
    }
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ── REPORT ────────────────────────────────────────────────────────────────────

def print_report(market: dict, rec: dict,
                 trade: dict, account: dict,
                 market_open: bool) -> None:
    sep    = "─" * 62
    action = rec["action"]
    icon   = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(action, "⚪")
    mkt    = "🟢 OPEN" if market_open else "🔴 CLOSED"

    print(f"\n{sep}")
    print(f"  {market['ticker']}  •  ${market['current_price']}  "
          f"({market['change_7d_pct']:+.1f}% / 7d)  "
          f"|  RSI: {market['rsi_14']}  |  MACD hist: {market['macd_hist']}  "
          f"|  Market: {mkt}")
    print(f"  {icon}  {action}  (Confidence: {rec['confidence']}/10)"
          f"  —  {rec.get('time_horizon', '')}")
    print(f"  {rec.get('reasoning', '')}")
    for risk in rec.get("key_risks", []):
        print(f"  ⚠  {risk}")
    print(f"  📋  {trade['note']}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    if not ALPACA_API_KEY or not ANTHROPIC_API_KEY:
        print("ERROR: Fill in your API keys.")
        return

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n{'═'*62}")
    print(f"  Claude AI Stock Advisor v2 — Alpaca Paper  •  {now}")
    print(f"{'═'*62}")

    try:
        account     = get_account()
        positions   = get_positions()
        market_open = is_market_open()
    except requests.HTTPError as e:
        print(f"Alpaca API error: {e}")
        return

    port_val = float(account.get("portfolio_value", 0))
    buying   = float(account.get("buying_power", 0))
    mkt_str  = "OPEN 🟢" if market_open else "CLOSED 🔴"
    print(f"\n  💼  Portfolio: ${port_val:,.2f}  "
          f"|  Buying power: ${buying:,.2f}")
    print(f"  📅  Market: {mkt_str}  |  Open positions: {len(positions)}")

    for ticker in TICKERS:
        print(f"\n  Fetching {ticker}...")
        try:
            market       = fetch_market_data(ticker)
            headlines    = fetch_news(ticker)
            politicians  = fetch_capitol_trades(ticker)
            prompt       = build_prompt(market, headlines, account, positions, politicians)

            print(f"  Calling Claude for {ticker}...")
            rec   = call_claude(prompt)
            trade = execute_trade(rec, market, positions, market_open)

            print_report(market, rec, trade, account, market_open)
            log_to_csv(market, rec, trade, account)

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
