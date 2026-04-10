"""
Coinbase Advanced Trade — AI Paper Trading Script v2
=====================================================
Uses Coinbase's newer EC private key (JWT) authentication format.

Requirements:
  pip install requests anthropic yfinance cryptography

Configuration:
  Fill in the three variables in the CONFIGURATION section below.
  COINBASE_API_KEY    = the full "name" field from your key JSON
  COINBASE_API_SECRET = the full "privateKey" field from your key JSON
  ANTHROPIC_API_KEY   = your Anthropic API key
"""

import os
import csv
import json
import time
import datetime
import requests
import yfinance as yf
import anthropic

# JWT signing imports (from cryptography package)
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend
import jwt  # from PyJWT — install with: pip install pyjwt

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

COINBASE_API_KEY    = "organizations/57955423-88eb-47b1-9c80-2f92067a236d/apiKeys/d4ce5a8f-8aa4-47f7-b567-7caedf487b3b"
COINBASE_API_SECRET = "-----BEGIN EC PRIVATE KEY-----\nMHcCAQEEIPUoA+1p6gVdymIS4J/IUrrEu0QDIt1kP9AsW4yPdXOFoAoGCCqGSM49\nAwEHoUQDQgAEcEcKk+1iwLi4P8K3pLNvC91RjdJHQOPm7/taD70ATQWXM3Mbrqn+\nusXbE9q3xjRkcD3LbrqQMC+UkLCHLJat/w==\n-----END EC PRIVATE KEY-----\n"

ANTHROPIC_API_KEY   = "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"

PRODUCT_ID      = "ETH-EUR"
NEWS_TICKER     = "ETH-EUR"
PAPER_BALANCE   = 1000.0
MAX_POSITION    = 0.15
LOG_FILE        = "coinbase_trade_log.csv"

# ── COINBASE JWT AUTH ─────────────────────────────────────────────────────────

COINBASE_BASE = "https://api.coinbase.com"

def _build_jwt(method: str, path: str) -> str:
    """Build a JWT token for Coinbase Advanced Trade API authentication."""
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
    token = jwt.encode(
        payload,
        private_key,
        algorithm="ES256",
        headers={"kid": COINBASE_API_KEY, "nonce": str(int(time.time() * 1000))},
    )
    return token


def _cb_get(path: str) -> dict:
    """Authenticated GET request to Coinbase Advanced Trade API."""
    token   = _build_jwt("GET", path.split("?")[0])
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
    }
    resp = requests.get(COINBASE_BASE + path, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()


# ── MARKET DATA ───────────────────────────────────────────────────────────────

def fetch_coinbase_candles(product_id: str, limit: int = 14) -> list[dict]:
    """Fetch daily OHLCV candles from Coinbase."""
    end   = int(time.time())
    start = end - 86400 * limit
    path  = (f"/api/v3/brokerage/products/{product_id}/candles"
             f"?start={start}&end={end}&granularity=ONE_DAY&limit={limit}")
    data    = _cb_get(path)
    candles = data.get("candles", [])
    candles.sort(key=lambda c: int(c.get("start", 0)))
    return candles


def fetch_coinbase_price(product_id: str) -> dict:
    """Fetch live best bid/ask for a product."""
    path = f"/api/v3/brokerage/best_bid_ask?product_ids={product_id}"
    data = _cb_get(path)
    books = data.get("pricebooks", [{}])
    book  = books[0] if books else {}
    bids  = book.get("bids", [{}])
    asks  = book.get("asks", [{}])
    bid   = float(bids[0].get("price", 0)) if bids else 0
    ask   = float(asks[0].get("price", 0)) if asks else 0
    mid   = round((bid + ask) / 2, 2) if bid and ask else 0
    return {"bid": bid, "ask": ask, "mid": mid}


def build_market_data(candles: list[dict], live_price: dict) -> dict:
    """Calculate technical indicators from candle data."""
    closes  = [float(c["close"])  for c in candles]
    volumes = [float(c["volume"]) for c in candles]
    dates   = [
        datetime.datetime.fromtimestamp(int(c["start"])).strftime("%Y-%m-%d")
        for c in candles
    ]

    sma7  = round(sum(closes[-7:])  / min(7,  len(closes)), 2)
    sma14 = round(sum(closes[-14:]) / min(14, len(closes)), 2)

    deltas   = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains    = [d for d in deltas if d > 0]
    losses   = [-d for d in deltas if d < 0]
    avg_gain = sum(gains[-14:])  / 14 if gains  else 0
    avg_loss = sum(losses[-14:]) / 14 if losses else 0.001
    rsi      = round(100 - (100 / (1 + avg_gain / avg_loss)), 1)

    price_now = live_price["mid"] or closes[-1]
    price_1w  = closes[-7] if len(closes) >= 7 else closes[0]
    change_1w = round(((price_now - price_1w) / price_1w) * 100, 2)

    return {
        "product":       PRODUCT_ID,
        "current_price": price_now,
        "bid":           live_price["bid"],
        "ask":           live_price["ask"],
        "change_7d_pct": change_1w,
        "sma_7":         sma7,
        "sma_14":        sma14,
        "rsi_14":        rsi,
        "avg_volume_7d": round(sum(volumes[-7:]) / 7, 4),
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

def build_prompt(market: dict, headlines: list[str]) -> str:
    price_table = "\n".join(
        f"  {date}: €{price}" for date, price in market["price_history"]
    )
    news_block = "\n".join(f"  - {h}" for h in headlines)
    return f"""You are a cautious quantitative trading analyst specialising in crypto markets.
Analyse the following live market data and provide a clear trading recommendation.

## Asset: {market['product']}

### Price data (last {len(market['price_history'])} days)
{price_table}

### Live order book
- Best bid: €{market['bid']}
- Best ask: €{market['ask']}
- Mid price: €{market['current_price']}

### Key indicators
- 7-day change:   {market['change_7d_pct']:+.1f}%
- SMA-7:          €{market['sma_7']}
- SMA-14:         €{market['sma_14']}
- RSI-14:         {market['rsi_14']} (oversold <30, overbought >70)
- Avg daily vol:  {market['avg_volume_7d']} ETH

### Recent news headlines
{news_block}

---

Respond with a JSON object in EXACTLY this format (no markdown fences):
{{
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": 1-10,
  "price_target": <number or null>,
  "stop_loss":    <number or null>,
  "reasoning":    "<2-3 sentence explanation>",
  "key_risks":    ["<risk 1>", "<risk 2>"],
  "time_horizon": "short-term (days)" | "medium-term (weeks)" | "long-term (months)"
}}

Be conservative. When in doubt, HOLD. This is paper trading."""


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


# ── PAPER TRADING ─────────────────────────────────────────────────────────────

_paper = {"eur": PAPER_BALANCE, "eth": 0.0, "avg_buy": 0.0}

def simulate_trade(action: str, confidence: int, price: float) -> dict:
    trade = {"action": action, "price": price,
             "eth_qty": 0.0, "eur_value": 0.0, "note": ""}

    if action == "BUY":
        max_eur = _paper["eur"] * MAX_POSITION * (confidence / 10)
        if max_eur < 1:
            trade["note"] = "Insufficient paper balance"
            return trade
        eth_qty = round(max_eur / price, 6)
        cost    = round(eth_qty * price, 2)
        total   = _paper["eth"] + eth_qty
        _paper["avg_buy"] = round(
            (_paper["avg_buy"] * _paper["eth"] + price * eth_qty) / total, 2
        ) if total > 0 else price
        _paper["eth"] = round(total, 6)
        _paper["eur"] = round(_paper["eur"] - cost, 2)
        trade.update({"eth_qty": eth_qty, "eur_value": cost})

    elif action == "SELL":
        if _paper["eth"] <= 0:
            trade["note"] = "No ETH held — nothing to sell"
            return trade
        eth_qty  = round(_paper["eth"] * (confidence / 10), 6)
        proceeds = round(eth_qty * price, 2)
        _paper["eth"] = round(_paper["eth"] - eth_qty, 6)
        _paper["eur"] = round(_paper["eur"] + proceeds, 2)
        trade.update({"eth_qty": eth_qty, "eur_value": proceeds})

    else:
        trade["note"] = "Holding position"

    return trade


# ── CSV LOGGING ───────────────────────────────────────────────────────────────

def log_to_csv(market: dict, rec: dict, trade: dict) -> None:
    file_exists = os.path.isfile(LOG_FILE)
    fieldnames  = [
        "timestamp", "product", "price", "bid", "ask",
        "change_7d_pct", "sma_7", "sma_14", "rsi_14",
        "action", "confidence", "time_horizon",
        "price_target", "stop_loss", "reasoning",
        "trade_eth", "trade_eur", "trade_note",
        "paper_eur", "paper_eth", "paper_avg_buy",
    ]
    row = {
        "timestamp":     datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "product":       market["product"],
        "price":         market["current_price"],
        "bid":           market["bid"],
        "ask":           market["ask"],
        "change_7d_pct": market["change_7d_pct"],
        "sma_7":         market["sma_7"],
        "sma_14":        market["sma_14"],
        "rsi_14":        market["rsi_14"],
        "action":        rec["action"],
        "confidence":    rec["confidence"],
        "time_horizon":  rec.get("time_horizon", ""),
        "price_target":  rec.get("price_target", ""),
        "stop_loss":     rec.get("stop_loss", ""),
        "reasoning":     rec.get("reasoning", ""),
        "trade_eth":     trade.get("eth_qty", 0),
        "trade_eur":     trade.get("eur_value", 0),
        "trade_note":    trade.get("note", ""),
        "paper_eur":     _paper["eur"],
        "paper_eth":     _paper["eth"],
        "paper_avg_buy": _paper["avg_buy"],
    }
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"  📁  Logged to {LOG_FILE}")


# ── REPORT ────────────────────────────────────────────────────────────────────

def print_report(market: dict, rec: dict, trade: dict) -> None:
    sep = "─" * 62
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    print(f"\n{'═'*62}")
    print(f"  Claude AI Trading Advisor — Coinbase  •  {now}")
    print(f"{'═'*62}")
    print(f"\n📊  {market['product']}  •  €{market['current_price']}  "
          f"({market['change_7d_pct']:+.1f}% / 7d)")
    print(f"    Bid: €{market['bid']}  |  Ask: €{market['ask']}")
    print(f"    SMA-7: €{market['sma_7']}  |  SMA-14: €{market['sma_14']}  "
          f"|  RSI-14: {market['rsi_14']}")

    action = rec["action"]
    icon   = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(action, "⚪")

    print(f"\n{sep}")
    print(f"  {icon}  RECOMMENDATION: {action}  "
          f"(Confidence: {rec['confidence']}/10)")
    print(f"{sep}")
    print(f"\n  Reasoning:\n  {rec['reasoning']}")
    if rec.get("price_target"):
        print(f"\n  Price target:  €{rec['price_target']}")
    if rec.get("stop_loss"):
        print(f"  Stop-loss:     €{rec['stop_loss']}")
    print(f"  Time horizon:  {rec.get('time_horizon', 'N/A')}")
    risks = rec.get("key_risks", [])
    if risks:
        print(f"\n  Key risks:")
        for r in risks:
            print(f"    • {r}")

    print(f"\n{sep}")
    if trade["action"] == "BUY":
        print(f"  📋  PAPER TRADE: Buy {trade['eth_qty']} ETH "
              f"@ €{trade['price']} = €{trade['eur_value']}")
    elif trade["action"] == "SELL":
        print(f"  📋  PAPER TRADE: " +
              (trade["note"] if trade["note"] else
               f"Sell {trade['eth_qty']} ETH @ €{trade['price']} = €{trade['eur_value']}"))
    else:
        print(f"  📋  PAPER TRADE: No trade — HOLD")

    print(f"\n  💼  Paper portfolio:")
    print(f"      EUR balance:  €{_paper['eur']}")
    print(f"      ETH held:     {_paper['eth']} ETH")
    if _paper["avg_buy"]:
        unreal = round((market["current_price"] - _paper["avg_buy"]) * _paper["eth"], 2)
        print(f"      Avg buy:      €{_paper['avg_buy']}")
        print(f"      Unrealised:   €{unreal:+.2f}")

    print(f"\n{'═'*62}")
    print("  ⚠  Paper trading only. Not financial advice.")
    print(f"{'═'*62}\n")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    if "YOUR-ORG-ID" in COINBASE_API_KEY or "YOUR_ANTHROPIC" in ANTHROPIC_API_KEY:
        print("ERROR: Fill in your API keys in the CONFIGURATION section at the top.")
        return

    print(f"Fetching live {PRODUCT_ID} data from Coinbase...")
    try:
        candles    = fetch_coinbase_candles(PRODUCT_ID, limit=14)
        live_price = fetch_coinbase_price(PRODUCT_ID)
    except requests.HTTPError as e:
        print(f"Coinbase API error: {e}")
        print("Check your API key, secret, and permissions.")
        return
    except Exception as e:
        print(f"Unexpected error fetching Coinbase data: {e}")
        return

    market    = build_market_data(candles, live_price)
    headlines = fetch_news(NEWS_TICKER)

    print("Calling Claude API for analysis...")
    prompt = build_prompt(market, headlines)
    rec    = call_claude(prompt)
    trade  = simulate_trade(rec["action"], rec["confidence"], market["current_price"])

    print_report(market, rec, trade)
    log_to_csv(market, rec, trade)


if __name__ == "__main__":
    main()
