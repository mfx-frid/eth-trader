"""
Coinbase Advanced Trade — AI Crypto Advisor v3
===============================================
Multi-ticker version — analyses multiple crypto pairs in one run.
Uses JWT authentication (EC private key format).

Requirements:
  pip install requests anthropic yfinance cryptography pyjwt

Configuration:
  Fill in the three variables in the CONFIGURATION section below.
"""

import os
import csv
import json
import time
import datetime
import requests
import yfinance as yf
import anthropic
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend
import jwt

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

COINBASE_API_KEY    = os.environ.get("COINBASE_API_KEY",    "")
COINBASE_API_SECRET = os.environ.get("COINBASE_API_SECRET", "")
ANTHROPIC_API_KEY   = os.environ.get("ANTHROPIC_API_KEY",   "")

# Crypto pairs to analyse — all must be available on Coinbase Advanced Trade
TICKERS = [
    "ETH-EUR",
    "BTC-EUR",
    "XRP-EUR",
    "SOL-EUR",
    "LTC-EUR",
    "DOGE-EUR",
]

# Yahoo Finance mapping for news headlines
YAHOO_NEWS_MAP = {
    "ETH-EUR":  "ETH-EUR",
    "BTC-EUR":  "BTC-USD",
    "XRP-EUR":  "XRP-USD",
    "SOL-EUR":  "SOL-USD",
    "LTC-EUR":  "LTC-USD",
    "DOGE-EUR": "DOGE-USD",
}

LOG_FILE        = "coinbase_trade_log.csv"
PAPER_BALANCE   = 1000.0   # Virtual EUR balance
MAX_POSITION    = 0.15     # Max 15% of balance per trade
MIN_CONFIDENCE  = 7        # Minimum confidence to BUY

COINBASE_BASE   = "https://api.coinbase.com"

# ── COINBASE JWT AUTH ─────────────────────────────────────────────────────────

def _build_jwt(method: str, path: str) -> str:
    private_key = serialization.load_pem_private_key(
        COINBASE_API_SECRET.encode("utf-8"),
        password=None,
        backend=default_backend()
    )
    payload = {
        "sub": COINBASE_API_KEY,
        "iss": "coinbase-cloud",
        "nbf": int(time.time()),
        "exp": int(time.time()) + 120,
        "uri": f"{method.upper()} api.coinbase.com{path}",
    }
    return jwt.encode(
        payload,
        private_key,
        algorithm="ES256",
        headers={"kid": COINBASE_API_KEY, "nonce": str(int(time.time() * 1000))},
    )


def _cb_get(path: str) -> dict:
    token   = _build_jwt("GET", path.split("?")[0])
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
    }
    resp = requests.get(COINBASE_BASE + path, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()


# ── MARKET DATA ───────────────────────────────────────────────────────────────

def fetch_candles(product_id: str, limit: int = 14) -> list[dict]:
    end   = int(time.time())
    start = end - 86400 * limit
    path  = (f"/api/v3/brokerage/products/{product_id}/candles"
             f"?start={start}&end={end}&granularity=ONE_DAY&limit={limit}")
    data    = _cb_get(path)
    candles = data.get("candles", [])
    candles.sort(key=lambda c: int(c.get("start", 0)))
    return candles


def fetch_live_price(product_id: str) -> dict:
    path = f"/api/v3/brokerage/best_bid_ask?product_ids={product_id}"
    data = _cb_get(path)
    books = data.get("pricebooks", [{}])
    book  = books[0] if books else {}
    bids  = book.get("bids", [{}])
    asks  = book.get("asks", [{}])
    bid   = float(bids[0].get("price", 0)) if bids else 0
    ask   = float(asks[0].get("price", 0)) if asks else 0
    mid   = round((bid + ask) / 2, 4) if bid and ask else 0
    return {"bid": bid, "ask": ask, "mid": mid}


def build_market_data(product_id: str,
                      candles: list[dict],
                      live: dict) -> dict:
    closes  = [float(c["close"])  for c in candles]
    volumes = [float(c["volume"]) for c in candles]
    dates   = [
        datetime.datetime.fromtimestamp(int(c["start"])).strftime("%Y-%m-%d")
        for c in candles
    ]

    sma7  = round(sum(closes[-7:])  / min(7,  len(closes)), 4)
    sma14 = round(sum(closes[-14:]) / min(14, len(closes)), 4)

    deltas   = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains    = [d for d in deltas if d > 0]
    losses   = [-d for d in deltas if d < 0]
    avg_gain = sum(gains[-14:])  / 14 if gains  else 0
    avg_loss = sum(losses[-14:]) / 14 if losses else 0.001
    rsi      = round(100 - (100 / (1 + avg_gain / avg_loss)), 1)

    price_now = live["mid"] or closes[-1]
    price_1w  = closes[-7] if len(closes) >= 7 else closes[0]
    change_1w = round(((price_now - price_1w) / price_1w) * 100, 2)

    return {
        "product":       product_id,
        "current_price": price_now,
        "bid":           live["bid"],
        "ask":           live["ask"],
        "change_7d_pct": change_1w,
        "sma_7":         sma7,
        "sma_14":        sma14,
        "rsi_14":        rsi,
        "avg_volume_7d": round(sum(volumes[-7:]) / 7, 4),
        "price_history": list(zip(dates, [round(c, 4) for c in closes])),
    }


def fetch_news(yahoo_ticker: str, max_items: int = 5) -> list[str]:
    try:
        stock     = yf.Ticker(yahoo_ticker)
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


# ── PAPER PORTFOLIO ───────────────────────────────────────────────────────────

# Persists across tickers within one run
_portfolio = {
    "eur_balance": PAPER_BALANCE,
    "holdings":    {},   # { "ETH-EUR": {"qty": 0.5, "avg_buy": 1800.0} }
}


def simulate_trade(action: str, confidence: int,
                   product: str, price: float) -> dict:
    trade = {"action": action, "product": product,
             "price": price, "qty": 0.0,
             "eur_value": 0.0, "note": ""}
    p = _portfolio

    if action == "BUY":
        if confidence < MIN_CONFIDENCE:
            trade["note"] = f"Confidence {confidence}/10 below threshold — skipped"
            return trade
        if p["eur_balance"] < 10:
            trade["note"] = "Insufficient paper balance"
            return trade
        if product in p["holdings"] and p["holdings"][product]["qty"] > 0:
            trade["note"] = "Already holding — skipped"
            return trade

        max_eur = p["eur_balance"] * MAX_POSITION * (confidence / 10)
        qty     = round(max_eur / price, 6)
        cost    = round(qty * price, 2)

        p["eur_balance"] -= cost
        p["holdings"][product] = {"qty": qty, "avg_buy": price}
        trade.update({"qty": qty, "eur_value": cost,
                      "note": "Paper buy executed ✅"})

    elif action == "SELL":
        holding = p["holdings"].get(product)
        if not holding or holding["qty"] <= 0:
            trade["note"] = "No position to sell — skipped"
            return trade
        qty      = holding["qty"]
        proceeds = round(qty * price, 2)
        pnl      = round(proceeds - (holding["avg_buy"] * qty), 2)
        p["eur_balance"] += proceeds
        p["holdings"][product] = {"qty": 0.0, "avg_buy": 0.0}
        trade.update({"qty": qty, "eur_value": proceeds,
                      "note": f"Paper sell executed ✅  P&L: €{pnl:+.2f}"})
    else:
        trade["note"] = "Holding — no trade"

    return trade


# ── CLAUDE ANALYSIS ───────────────────────────────────────────────────────────

def build_prompt(market: dict, headlines: list[str]) -> str:
    price_table = "\n".join(
        f"  {date}: €{price}" for date, price in market["price_history"]
    )
    news_block = "\n".join(f"  - {h}" for h in headlines)
    holding    = _portfolio["holdings"].get(market["product"])
    pos_info   = (
        f"Currently holding {holding['qty']} units "
        f"(avg buy: €{holding['avg_buy']:.4f})"
        if holding and holding["qty"] > 0
        else "No current position."
    )

    return f"""You are a cautious quantitative crypto trading analyst.
Analyse the following data and provide a clear trading recommendation.

## Asset: {market['product']}

### Price history (last {len(market['price_history'])} days)
{price_table}

### Live order book
- Best bid: €{market['bid']}
- Best ask: €{market['ask']}
- Mid:       €{market['current_price']}

### Indicators
- 7-day change: {market['change_7d_pct']:+.1f}%
- SMA-7:        €{market['sma_7']}
- SMA-14:       €{market['sma_14']}
- RSI-14:       {market['rsi_14']} (oversold <30, overbought >70)
- Avg vol/day:  {market['avg_volume_7d']}

### Current position
{pos_info}

### Paper portfolio
- EUR balance: €{round(_portfolio['eur_balance'], 2)}

### Recent news
{news_block}

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
- SELL only if currently holding
- When in doubt, HOLD
- This is paper trading"""


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


# ── CSV LOGGING ───────────────────────────────────────────────────────────────

def log_to_csv(market: dict, rec: dict, trade: dict) -> None:
    file_exists = os.path.isfile(LOG_FILE)
    fieldnames  = [
        "timestamp", "product", "price", "bid", "ask",
        "change_7d_pct", "sma_7", "sma_14", "rsi_14",
        "action", "confidence", "time_horizon",
        "price_target", "stop_loss", "reasoning",
        "trade_qty", "trade_eur", "trade_note",
        "paper_eur_balance",
    ]
    row = {
        "timestamp":       datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "product":         market["product"],
        "price":           market["current_price"],
        "bid":             market["bid"],
        "ask":             market["ask"],
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
        "trade_qty":       trade.get("qty", 0),
        "trade_eur":       trade.get("eur_value", 0),
        "trade_note":      trade.get("note", ""),
        "paper_eur_balance": round(_portfolio["eur_balance"], 2),
    }
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ── REPORT ────────────────────────────────────────────────────────────────────

def print_ticker_report(market: dict, rec: dict, trade: dict) -> None:
    action = rec["action"]
    icon   = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(action, "⚪")
    sep    = "─" * 62

    print(f"\n{sep}")
    print(f"  {market['product']}  •  €{market['current_price']}  "
          f"({market['change_7d_pct']:+.1f}% / 7d)  "
          f"|  RSI: {market['rsi_14']}")
    print(f"  Bid: €{market['bid']}  |  Ask: €{market['ask']}  "
          f"|  SMA7: €{market['sma_7']}  |  SMA14: €{market['sma_14']}")
    print(f"  {icon}  {action}  (Confidence: {rec['confidence']}/10)"
          f"  —  {rec.get('time_horizon', '')}")
    print(f"  {rec.get('reasoning', '')}")
    if rec.get("stop_loss"):
        print(f"  Stop-loss: €{rec['stop_loss']}")
    if rec.get("price_target"):
        print(f"  Target:    €{rec['price_target']}")
    for risk in rec.get("key_risks", []):
        print(f"  ⚠  {risk}")
    print(f"  📋  {trade['note']}")
    if trade.get("qty") and trade["qty"] > 0:
        side = "Buy" if action == "BUY" else "Sell"
        print(f"      {side} {trade['qty']} units @ €{trade['price']}"
              f" = €{trade['eur_value']}")


def print_portfolio_summary() -> None:
    sep = "═" * 62
    p   = _portfolio
    print(f"\n{sep}")
    print(f"  💼  Paper portfolio summary")
    print(f"{sep}")
    print(f"  EUR balance: €{round(p['eur_balance'], 2)}")
    for product, h in p["holdings"].items():
        if h["qty"] > 0:
            print(f"  {product}: {h['qty']} units  "
                  f"(avg buy: €{h['avg_buy']:.4f})")
    if not any(h["qty"] > 0 for h in p["holdings"].values()):
        print(f"  No open positions")
    print(f"{sep}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    if not COINBASE_API_KEY or not ANTHROPIC_API_KEY:
        print("ERROR: Fill in your API keys in the CONFIGURATION section.")
        return

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n{'═'*62}")
    print(f"  Claude AI Crypto Advisor v3 — Coinbase  •  {now}")
    print(f"  Tracking: {', '.join(TICKERS)}")
    print(f"{'═'*62}")

    for product_id in TICKERS:
        print(f"\n  Fetching {product_id}...")
        try:
            candles    = fetch_candles(product_id, limit=14)
            live       = fetch_live_price(product_id)
            market     = build_market_data(product_id, candles, live)
            yahoo_tick = YAHOO_NEWS_MAP.get(product_id, product_id)
            headlines  = fetch_news(yahoo_tick)

            print(f"  Calling Claude for {product_id}...")
            prompt = build_prompt(market, headlines)
            rec    = call_claude(prompt)
            trade  = simulate_trade(
                rec["action"], rec["confidence"],
                product_id, market["current_price"]
            )

            print_ticker_report(market, rec, trade)
            log_to_csv(market, rec, trade)

        except requests.HTTPError as e:
            print(f"  Coinbase API error for {product_id}: {e}")
            continue
        except Exception as e:
            print(f"  Error processing {product_id}: {e}")
            continue

    print_portfolio_summary()
    print(f"\n  📁  Logged to {LOG_FILE}")
    print(f"  ⚠  Paper trading only. Not financial advice.\n")


if __name__ == "__main__":
    main()
