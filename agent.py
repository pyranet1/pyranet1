#!/usr/bin/env python3
"""
Watchlist buy-signal agent.

Mirrors the "Intrinsic Value, DCF & Buy Signal" TradingView indicator:
  - Graham Formula, Graham Number and a 2-stage free-cash-flow DCF
  - Buy zone  = price at least MIN_MOS % below intrinsic value
  - Buy signal = buy zone + timing confirmation (RSI recovery / EMA cross / zone entry)

Run once per day after the market closes (see README). Alerts go to Telegram
and/or email, whichever you configure through environment variables.

Usage:
  python agent.py              # scan watchlist, send alerts for new signals
  python agent.py --dry-run    # scan and print only, send nothing
  python agent.py --test-alert # send a test message to check your setup
"""
import argparse
import json
import math
import os
import smtplib
import sys
from datetime import datetime, timedelta
from email.message import EmailMessage
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
WATCHLIST_FILE = HERE / "watchlist.txt"
STATE_FILE = HERE / "state.json"          # remembers when each symbol last alerted

# ───────────── SETTINGS (same defaults as the Pine script) ─────────────
CFG = {
    "method": "all",            # graham_formula | graham_number | dcf | graham_avg | all
    "aaa_yield": 4.5,           # AAA corporate bond yield, % (update periodically)
    "manual_growth": 8.0,       # % used when EPS growth cannot be computed
    "max_growth": 15.0,         # cap for auto-calculated growth, %
    "dcf_growth_same": True,    # True: DCF uses the Graham growth; False: use dcf_growth
    "dcf_growth": 8.0,
    "dcf_years": 10,
    "discount": 10.0,           # required return, %
    "term_growth": 2.5,         # terminal growth, %
    "min_mos": 25.0,            # minimum margin of safety, %
    "max_pe": None,             # e.g. 25 to require PE <= 25, None = off
    "confirm": "rsi",           # rsi | ema | enter
    "rsi_len": 14,
    "rsi_level": 40.0,
    "ema_len": 50,
    "cooldown_days": 28,        # don't re-alert the same symbol within this window
}


# ───────────── VALUATION MATH ─────────────
def graham_formula(eps, g, y):
    return eps * (8.5 + 2 * g) * 4.4 / y


def graham_number(eps, bvps):
    return math.sqrt(22.5 * eps * bvps)


def dcf_value(fcfps, g, r, tg, n):
    """Two-stage DCF per share: n years of growth g, then Gordon terminal value."""
    if fcfps is None or fcfps <= 0 or r <= tg:
        return None
    q = (1 + g / 100) / (1 + r / 100)
    stage = n if abs(q - 1) < 1e-9 else q * (1 - q ** n) / (1 - q)
    fcf_n = fcfps * (1 + g / 100) ** n
    tv = fcf_n * (1 + tg / 100) / ((r - tg) / 100)
    return fcfps * stage + tv / (1 + r / 100) ** n


def compute_intrinsic(d, cfg):
    """Returns (intrinsic_value_or_None, dict_of_method_values, growth_used)."""
    growth = d.get("growth")
    g = cfg["manual_growth"] if growth is None else min(max(growth, 0.0), cfg["max_growth"])
    eps, bv = d.get("eps"), d.get("bvps")
    vals = {}
    if eps and eps > 0:
        vals["graham_formula"] = graham_formula(eps, g, cfg["aaa_yield"])
        if bv and bv > 0:
            vals["graham_number"] = graham_number(eps, bv)
    fcf, shares = d.get("fcf"), d.get("shares")
    fcfps = fcf / shares if fcf and shares and shares > 0 else None
    dg = g if cfg["dcf_growth_same"] else cfg["dcf_growth"]
    dv = dcf_value(fcfps, dg, cfg["discount"], cfg["term_growth"], cfg["dcf_years"])
    if dv is not None:
        vals["dcf"] = dv

    m = cfg["method"]
    if m in ("graham_formula", "graham_number", "dcf"):
        chosen = [vals[m]] if m in vals else []
    elif m == "graham_avg":
        chosen = [vals[k] for k in ("graham_formula", "graham_number") if k in vals]
    else:
        chosen = list(vals.values())
    iv = sum(chosen) / len(chosen) if chosen else None
    return iv, vals, g


# ───────────── INDICATORS ─────────────
def rsi(close, length):
    """Wilder RSI (same smoothing as TradingView's ta.rsi)."""
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    rs = up / down
    return 100 - 100 / (1 + rs)


def ema(close, length):
    return close.ewm(span=length, adjust=False).mean()


# ───────────── SIGNAL LOGIC ─────────────
def evaluate(symbol, d, cfg):
    close = d["close"].dropna()
    if len(close) < max(cfg["ema_len"], cfg["rsi_len"]) + 2:
        return {"symbol": symbol, "status": "not enough price history"}

    iv, vals, g = compute_intrinsic(d, cfg)
    if iv is None or iv <= 0:
        return {"symbol": symbol, "status": "no valid valuation (negative/missing EPS and FCF?)"}

    price, prev_price = float(close.iloc[-1]), float(close.iloc[-2])
    eps = d.get("eps")
    pe = price / eps if eps and eps > 0 else None
    prev_pe = prev_price / eps if eps and eps > 0 else None

    mos = (iv - price) / iv * 100
    prev_mos = (iv - prev_price) / iv * 100

    def pe_ok(p):
        return cfg["max_pe"] is None or (p is not None and p <= cfg["max_pe"])

    in_zone = mos >= cfg["min_mos"] and pe_ok(pe)
    prev_zone = prev_mos >= cfg["min_mos"] and pe_ok(prev_pe)

    r = rsi(close, cfg["rsi_len"])
    e = ema(close, cfg["ema_len"])
    rsi_cross = r.iloc[-2] <= cfg["rsi_level"] < r.iloc[-1]
    ema_cross = close.iloc[-2] <= e.iloc[-2] and close.iloc[-1] > e.iloc[-1]
    entered = in_zone and not prev_zone
    confirm = {"rsi": rsi_cross, "ema": ema_cross}.get(cfg["confirm"], entered)

    return {
        "symbol": symbol, "status": "ok", "price": price, "intrinsic": iv,
        "methods": vals, "growth": g, "pe": pe, "mos": mos, "rsi": float(r.iloc[-1]),
        "in_zone": bool(in_zone), "signal": bool(in_zone and confirm),
        "date": close.index[-1],
    }


# ───────────── DATA (Yahoo Finance) ─────────────
def fetch(symbol):
    import yfinance as yf  # imported here so the math above can be tested offline

    t = yf.Ticker(symbol)
    info = t.info or {}
    hist = t.history(period="2y", interval="1d", auto_adjust=False)

    # Yahoo sometimes reports financials in a different currency than the price
    # (some ADRs / dual listings). Values would then be meaningless, so skip them.
    pc, fc = info.get("currency"), info.get("financialCurrency")
    if pc and fc and pc != fc:
        raise ValueError(f"price currency {pc} != financial currency {fc}; skipped")

    growth = None
    try:
        inc = t.income_stmt
        row = inc.loc["Diluted EPS"].dropna()          # newest year first
        if len(row) >= 2 and row.iloc[0] > 0 and row.iloc[1] > 0:
            growth = (row.iloc[0] / row.iloc[1] - 1) * 100
    except Exception:
        pass

    return {
        "close": hist["Close"],
        "eps": info.get("trailingEps"),
        "bvps": info.get("bookValue"),
        "fcf": info.get("freeCashflow"),
        "shares": info.get("sharesOutstanding"),
        "growth": growth,
    }


# ───────────── ALERTING ─────────────
def send_telegram(text):
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not (token and chat):
        return False
    resp = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                         data={"chat_id": chat, "text": text}, timeout=20)
    resp.raise_for_status()
    return True


def send_email(subject, text):
    host, to = os.getenv("SMTP_HOST"), os.getenv("EMAIL_TO")
    user, pwd = os.getenv("SMTP_USER"), os.getenv("SMTP_PASS")
    if not (host and to and user and pwd):
        return False
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, user, to
    msg.set_content(text)
    with smtplib.SMTP(host, int(os.getenv("SMTP_PORT") or 587), timeout=30) as s:
        s.starttls()
        s.login(user, pwd)
        s.send_message(msg)
    return True


def notify(subject, text):
    sent = []
    for name, fn in (("Telegram", lambda: send_telegram(text)),
                     ("Email", lambda: send_email(subject, text))):
        try:
            if fn():
                sent.append(name)
        except Exception as exc:
            print(f"  ! {name} failed: {exc}", file=sys.stderr)
    return sent


def format_signal(r):
    m = ", ".join(f"{k}={v:.2f}" for k, v in r["methods"].items())
    pe = f"{r['pe']:.1f}" if r["pe"] else "n/a"
    return (f"BUY SIGNAL: {r['symbol']}\n"
            f"Price {r['price']:.2f} | Intrinsic {r['intrinsic']:.2f} | "
            f"Margin of safety {r['mos']:.1f}%\n"
            f"PE {pe} | RSI {r['rsi']:.0f} | Growth used {r['growth']:.1f}%\n"
            f"Methods: {m}")


# ───────────── STATE (cooldown) ─────────────
def load_state():
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def in_cooldown(state, symbol, days):
    last = state.get(symbol)
    if not last:
        return False
    return datetime.now() - datetime.fromisoformat(last) < timedelta(days=days)


# ───────────── EXPLAIN / REPLAY (for checking the logic) ─────────────
def explain(symbol, d, cfg, bars=10):
    """Print every input and step behind the signal for one symbol, plus a replay
    of the last ~2 years showing when it WOULD have fired."""
    close = d["close"].dropna()
    iv, vals, g = compute_intrinsic(d, cfg)
    eps, fcf, shares = d.get("eps"), d.get("fcf"), d.get("shares")
    fcfps = fcf / shares if fcf and shares and shares > 0 else None
    f = lambda v: "n/a" if v is None else f"{v:,.2f}"

    print(f"\n=== {symbol} ===")
    print("INPUTS (from Yahoo Finance)")
    print(f"  EPS (TTM): {f(eps)} | Book value/share: {f(d.get('bvps'))}")
    print(f"  Free cash flow: {f(fcf)} | Shares: {f(shares)} | FCF/share: {f(fcfps)}")
    gsrc = "auto, last FY EPS growth" if d.get("growth") is not None else "manual fallback"
    print(f"  Growth used: {g:.1f}% ({gsrc})")
    print("VALUATION")
    for k in ("graham_formula", "graham_number", "dcf"):
        print(f"  {k:<15}: {f(vals.get(k))}")
    print(f"  Chosen method '{cfg['method']}' -> intrinsic value: {f(iv)}")
    if iv is None or iv <= 0:
        print("  No valid intrinsic value, so no signal is possible for this stock.")
        return

    mos = (iv - close) / iv * 100
    pe = close / eps if eps and eps > 0 else pd.Series(float("nan"), index=close.index)
    pe_ok = pd.Series(True, index=close.index) if cfg["max_pe"] is None else (pe <= cfg["max_pe"])
    zone = (mos >= cfg["min_mos"]) & pe_ok
    r, e = rsi(close, cfg["rsi_len"]), ema(close, cfg["ema_len"])
    rsi_cross = (r.shift(1) <= cfg["rsi_level"]) & (r > cfg["rsi_level"])
    ema_cross = (close.shift(1) <= e.shift(1)) & (close > e)
    entered = zone & ~zone.shift(1, fill_value=False)
    confirm = {"rsi": rsi_cross, "ema": ema_cross}.get(cfg["confirm"], entered)
    signal = zone & confirm

    print(f"LAST {bars} DAILY BARS  (zone = margin of safety >= {cfg['min_mos']}%; confirm mode = {cfg['confirm']})")
    print(f"  {'Date':<11}{'Close':>9}{'MoS %':>8}{'RSI':>6}  Zone  Confirm  Signal")
    for dt in close.index[-bars:]:
        print(f"  {str(dt.date()):<11}{close[dt]:>9.2f}{mos[dt]:>8.1f}{r[dt]:>6.0f}"
              f"  {'yes' if zone[dt] else '-':<5} {'yes' if confirm[dt] else '-':<8} {'BUY' if signal[dt] else '-'}")

    fired, last = [], None
    for dt in signal[signal].index:
        if last is None or (dt - last).days >= cfg["cooldown_days"]:
            fired.append(dt)
            last = dt
    print("REPLAY (uses TODAY's intrinsic value on past prices, so it is an approximation)")
    print(f"  Days in buy zone: {int(zone.sum())} of {len(zone)} | Signals after cooldown: {len(fired)}")
    for dt in fired[-10:]:
        print(f"    {dt.date()}  close {close[dt]:.2f}  MoS {mos[dt]:.1f}%  RSI {r[dt]:.0f}")

    check = evaluate(symbol, d, cfg)
    ok = check.get("status") == "ok" and bool(check["signal"]) == bool(signal.iloc[-1])
    print(f"CROSS-CHECK: explain mode and the daily scan agree on the latest bar: {'yes' if ok else 'NO - report this'}")


# ───────────── MAIN ─────────────
def read_watchlist():
    lines = WATCHLIST_FILE.read_text().splitlines()
    return [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print only, send nothing")
    ap.add_argument("--test-alert", action="store_true", help="send a test message")
    ap.add_argument("--explain", metavar="SYMBOL", help="show the full working for one symbol")
    args = ap.parse_args()

    if args.explain:
        explain(args.explain, fetch(args.explain), CFG)
        return

    if args.test_alert:
        sent = notify("Buy-signal agent test", "Test alert: your buy-signal agent is set up correctly.")
        print("Sent via:", ", ".join(sent) if sent else "nothing (no Telegram/email configured)")
        return

    state, new_signals = load_state(), []
    print(f"{'Symbol':<10}{'Price':>10}{'Intrinsic':>11}{'MoS %':>8}{'PE':>7}{'RSI':>6}  Status")
    for sym in read_watchlist():
        try:
            r = evaluate(sym, fetch(sym), CFG)
        except Exception as exc:
            print(f"{sym:<10} error: {exc}")
            continue
        if r["status"] != "ok":
            print(f"{sym:<10} {r['status']}")
            continue
        tag = "BUY SIGNAL" if r["signal"] else ("buy zone" if r["in_zone"] else "wait")
        pe = f"{r['pe']:.1f}" if r["pe"] else "n/a"
        print(f"{sym:<10}{r['price']:>10.2f}{r['intrinsic']:>11.2f}{r['mos']:>8.1f}{pe:>7}{r['rsi']:>6.0f}  {tag}")
        if r["signal"] and not in_cooldown(state, sym, CFG["cooldown_days"]):
            new_signals.append(r)

    if not new_signals:
        print("\nNo new buy signals.")
        return

    body = "\n\n".join(format_signal(r) for r in new_signals)
    body += "\n\nScreening aid only, not financial advice. Verify the fundamentals before trading."
    print("\n" + body)
    if args.dry_run:
        print("\n(dry run: nothing sent, state not updated)")
        return
    sent = notify(f"Buy signal: {', '.join(r['symbol'] for r in new_signals)}", body)
    if sent:
        for r in new_signals:
            state[r["symbol"]] = datetime.now().isoformat()
        STATE_FILE.write_text(json.dumps(state, indent=2))
        print("Alert sent via:", ", ".join(sent))
    else:
        print("No alert channel configured, so nothing was sent (set Telegram or email variables).")


if __name__ == "__main__":
    main()
