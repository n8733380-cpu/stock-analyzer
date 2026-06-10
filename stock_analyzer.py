import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import requests
import json, os, re, uuid as _uuid
import time as _time

@st.cache_data(ttl=3600)
def _fetch_institutional_data():
    """從 TWSE T86 抓最近 5 個交易日外資買賣超，回傳 {code: {'consec': N, 'total': X(股)}}"""
    from datetime import datetime as _dt, timedelta as _td
    daily = {}
    d = _dt.today()
    while len(daily) < 5 and (_dt.today() - d).days < 60:
        d -= _td(days=1)
        ds = d.strftime("%Y%m%d")
        try:
            url = (f"https://www.twse.com.tw/fund/T86"
                   f"?response=json&date={ds}&selectType=ALLBUT0999")
            r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            data = r.json()
            if data.get("stat") == "OK":
                out = {}
                for row in data["data"]:
                    code = row[0].strip()
                    if len(code) == 4 and code.isdigit():
                        try:
                            out[code] = int(row[4].replace(",", "").strip())
                        except Exception:
                            pass
                daily[ds] = out
        except Exception:
            pass
        _time.sleep(0.3)
    dates = sorted(daily.keys())
    all_codes = set().union(*[set(v.keys()) for v in daily.values()])
    result = {}
    for code in all_codes:
        consec = 0
        total = 0
        for date in reversed(dates):
            net = daily[date].get(code, 0)
            if net > 0:
                consec += 1
                total += net
            else:
                break
        if consec > 0:
            result[code] = {"consec": consec, "total": total}
    return result


st.set_page_config(page_title="台股技術分析", layout="wide")
st.title("台股均線分析儀表板")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("設定")
    exchange = st.radio("交易所", ["上市 (.TW)", "上櫃 (.TWO)"], horizontal=True)
    suffix = ".TW" if "TW)" in exchange else ".TWO"

    stocks_input = st.text_input(
        "股票代號（逗號分隔）",
        value="2330, 2454, 0050",
        help="上市直接輸數字，例：2330, 2454, 0050"
    )
    period = st.selectbox("K棒期間", ["3mo", "6mo", "1y", "2y"], index=2)

    st.divider()
    st.subheader("均線交叉訊號")
    fast_ma = st.selectbox("快線（決定入出場）", [5, 10, 20], index=0)
    slow_ma = st.selectbox("慢線", [10, 20, 60], index=1)
    if fast_ma >= slow_ma:
        st.warning("快線應小於慢線")

    st.divider()
    st.subheader("止損設定")
    use_atr_stop = st.toggle("ATR 止損（取代固定%）", value=False,
                              help="以 ATR(14)×倍數動態計算，比固定%更貼近波動度")
    if use_atr_stop:
        atr_mult = st.slider("ATR 倍數", 1.0, 3.0, 2.0, step=0.5)
        stop_pct = 7
    else:
        stop_pct = st.slider("止損幅度 %", 2, 20, 7,
                             help="黃金交叉入場收盤價往下算，橘色虛線顯示在圖上")
        atr_mult = 2.0

    st.divider()
    st.subheader("指標顯示")
    show_bb = st.toggle("Bollinger Bands (20,2)", value=True)
    momentum_ind = st.radio("動能指標", ["RSI", "KD", "MACD", "OBV"], horizontal=True)

    st.divider()
    st.subheader("支撐偵測")
    pivot_window = st.slider("偵測視窗（日）", 3, 15, 7,
                             help="抓樞紐低點的左右範圍，越大找越長期的底部")
    support_lookback = st.slider("往回找幾天", 60, 240, 120)


# ── 工具函式 ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800)
def fetch(symbol, period):
    df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def calc_indicators(df, fast_ma, slow_ma, pivot_window, support_lookback):
    df = df.copy()
    for ma in [5, 10, 20, 60]:
        df[f"MA{ma}"] = df["Close"].rolling(ma).mean()

    fast_col = f"MA{fast_ma}"
    slow_col = f"MA{slow_ma}"
    df["gap"] = df[fast_col] - df[slow_col]
    df["gap_prev"] = df["gap"].shift(1)
    df["signal"] = 0
    df.loc[(df["gap"] > 0) & (df["gap_prev"] <= 0), "signal"] = 1
    df.loc[(df["gap"] < 0) & (df["gap_prev"] >= 0), "signal"] = -1

    # OBV 能量潮
    direction   = np.sign(df["Close"].diff().fillna(0))
    df["OBV"]   = (direction * df["Volume"]).cumsum()
    df["OBV_MA20"] = df["OBV"].rolling(20).mean()

    # 動能報酬（1M/3M/6M）
    for n, label in [(21, "R1M"), (63, "R3M"), (126, "R6M")]:
        df[label] = (df["Close"] / df["Close"].shift(n) - 1) * 100

    # RSI(14)
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI14"] = 100 - 100 / (1 + gain / (loss + 1e-9))

    # ATR(14)
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"]  - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    # ADX(14)
    _hdiff = df["High"].diff()
    _ldiff = -df["Low"].diff()
    _plus_dm  = _hdiff.where((_hdiff > _ldiff) & (_hdiff > 0), 0.0)
    _minus_dm = _ldiff.where((_ldiff > _hdiff) & (_ldiff > 0), 0.0)
    _atr_sm   = df["ATR14"] + 1e-9
    _plus_di  = 100 * (_plus_dm.rolling(14).mean()  / _atr_sm)
    _minus_di = 100 * (_minus_dm.rolling(14).mean() / _atr_sm)
    _dx = 100 * (_plus_di - _minus_di).abs() / (_plus_di + _minus_di + 1e-9)
    df["ADX14"] = _dx.rolling(14).mean()

    if df.empty:
        return df, fast_col, slow_col, []
    cutoff = df.index[-1] - pd.Timedelta(days=support_lookback)
    raw_pivots = _detect_pivot_lows(df[df.index >= cutoff]["Low"], pivot_window)
    supports = _cluster_levels(raw_pivots)

    return df, fast_col, slow_col, supports


def _detect_pivot_lows(low_series, window):
    arr = low_series.values
    idx = low_series.index
    pivots = []
    for i in range(window, len(arr) - window):
        seg = arr[i - window: i + window + 1]
        if arr[i] <= seg.min() + 1e-9:
            pivots.append((idx[i], float(arr[i])))
    return pivots


def _cluster_levels(pivots, pct=0.02):
    if not pivots:
        return []
    prices = sorted(p for _, p in pivots)
    clusters, group = [], [prices[0]]
    for p in prices[1:]:
        if group[-1] > 0 and abs(p - group[-1]) / group[-1] < pct:
            group.append(p)
        else:
            clusters.append(float(np.mean(group)))
            group = [p]
    clusters.append(float(np.mean(group)))
    return clusters


# ── 圖表指標計算 ──────────────────────────────────────────────────────────────

def _calc_bb(close, period=20, std_mult=2):
    c = pd.Series(close)
    mid  = c.rolling(period).mean()
    band = c.rolling(period).std()
    return mid.values, (mid + std_mult * band).values, (mid - std_mult * band).values


def _calc_kd(high, low, close, n=9, m=3):
    h   = pd.Series(high).rolling(n).max()
    l   = pd.Series(low).rolling(n).min()
    rsv = (pd.Series(close) - l) / (h - l + 1e-9) * 100
    k   = rsv.ewm(com=m - 1, adjust=False).mean()
    d   = k.ewm(com=m - 1, adjust=False).mean()
    return k.values, d.values


def _calc_macd(close, fast=12, slow=26, signal=9):
    c        = pd.Series(close)
    ema_fast = c.ewm(span=fast, adjust=False).mean()
    ema_slow = c.ewm(span=slow, adjust=False).mean()
    macd     = ema_fast - ema_slow
    sig      = macd.ewm(span=signal, adjust=False).mean()
    return macd.values, sig.values, (macd - sig).values


def calc_score(df, patterns, fast_col, slow_col, rs_val=None, revenue_yoy=None):
    """
    綜合評分 0-100。
    核心邏輯：訊號時機（35分）主導分數，量能確認（20分）次之，
    其餘為輔助加減分。空頭排列直接回傳 0；頂背離或過度延伸上限 40。
    """
    latest = df.iloc[-1]
    close  = float(latest["Close"])

    # 空頭排列直接 0
    if float(latest[fast_col]) <= float(latest[slow_col]):
        return 0

    top_divs = [p for p in patterns if "頂背離" in p]
    rsi_now  = float(latest["RSI14"]) if "RSI14" in df.columns and pd.notna(latest.get("RSI14")) else 50
    r1m      = float(df["R1M"].iloc[-1]) if "R1M" in df.columns and pd.notna(df["R1M"].iloc[-1]) else 0

    score = 0

    # ── 訊號時機（35分）：黃金交叉距今幾天，越近越高分 ──────────────
    sig_df = df[df["signal"] == 1]
    if not sig_df.empty:
        days_since = (df.index[-1] - sig_df.iloc[-1].name).days
        if days_since <= 2:
            score += 35
        elif days_since <= 5:
            score += 28
        elif days_since <= 10:
            score += 18
        elif days_since <= 20:
            score += 10
        elif days_since <= 45:
            score += 4

    # ── 量能確認（20分）────────────────────────────────────────────
    vol_avg = df["Volume"].rolling(20).mean().iloc[-1]
    if pd.notna(vol_avg) and float(vol_avg) > 0:
        vol_ratio = float(latest["Volume"]) / float(vol_avg)
        if vol_ratio >= 2.0:
            score += 20
        elif vol_ratio >= 1.5:
            score += 14
        elif vol_ratio >= 1.0:
            score += 7
        elif vol_ratio >= 0.7:
            score += 2

    # ── 型態訊號（20分）────────────────────────────────────────────
    pos_pats = [p for p in patterns if "頂背離" not in p]
    PAT_W = {"VCP": 12, "杯柄": 10, "雙底": 9, "平台底": 7, "OBV底背離": 8, "RSI底背離": 5}
    pat_score = sum(w for k, w in PAT_W.items() if any(k in p for p in pos_pats))
    score += min(20, pat_score)

    # 頂背離重罰（每個 -20）
    score -= len(top_divs) * 20

    # VCP/杯柄 突破時高 RSI 是動能確認，不是過度延伸
    _breakout_pat = any(k in p for k in ("VCP", "杯柄") for p in pos_pats)

    # ── RSI 位置（10分）────────────────────────────────────────────
    if 52 <= rsi_now <= 65:
        score += 10
    elif (45 <= rsi_now < 52) or (65 < rsi_now <= 72):
        score += 5
    elif rsi_now > 75 and not _breakout_pat:
        score -= 10

    # ── 1M 動能品質（10分）─────────────────────────────────────────
    if 3 < r1m <= 15:
        score += 10
    elif 0 < r1m <= 3 or 15 < r1m <= 25:
        score += 5
    elif r1m > 30:
        score -= 8

    # ── 相對強度 vs 0050（10分）────────────────────────────────────
    if rs_val is not None:
        if rs_val >= 15:
            score += 10
        elif rs_val >= 5:
            score += 6
        elif rs_val >= 0:
            score += 2
        elif rs_val <= -15:
            score -= 12
        elif rs_val <= -5:
            score -= 6

    # ── 52週位置（5分）─────────────────────────────────────────────
    high_52 = df["High"].tail(252).max() if len(df) >= 150 else df["High"].max()
    if float(high_52) > 0:
        dist_pct = (close / float(high_52) - 1) * 100
        if -8 <= dist_pct <= 0:
            score += 5   # 接近前高，突破點
        elif -20 <= dist_pct < -8:
            score += 3
        elif dist_pct < -40:
            score -= 3

    # ── MA20 > MA60 長期多頭（+5）──────────────────────────────────────────
    if "MA60" in df.columns and pd.notna(df["MA60"].iloc[-1]):
        if pd.notna(df[slow_col].iloc[-1]) and float(df[slow_col].iloc[-1]) > float(df["MA60"].iloc[-1]):
            score += 5

    # ── 月營收 YoY 加分 ─────────────────────────────────────────────────
    if revenue_yoy is not None:
        if revenue_yoy >= 30:    score += 15
        elif revenue_yoy >= 15:  score += 8
        elif revenue_yoy >= 5:   score += 3
        elif revenue_yoy <= -20: score -= 8

    # 頂背離或過度延伸 → 分數上限 40（VCP/杯柄 突破免受 RSI 上限限制）
    cap = 40 if (top_divs or (rsi_now >= 78 and not _breakout_pat) or r1m > 30) else 100

    return max(0, min(cap, score))


def _calc_max_drawdown(df):
    """計算圖表期間最大回撤"""
    close = df["Close"].values.astype(float)
    peak = close[0]
    max_dd = 0.0
    for p in close:
        if p > peak:
            peak = p
        dd = (peak - p) / peak
        if dd > max_dd:
            max_dd = dd
    return round(max_dd * 100, 1)


def _calc_momentum(df):
    """近 1M / 3M / 6M 報酬率"""
    close = df["Close"]
    latest = float(close.iloc[-1])
    result = {}
    for n, key in [(21, "R1M"), (63, "R3M"), (126, "R6M")]:
        if len(close) > n:
            base = float(close.iloc[-1 - n])
            result[key] = round((latest / base - 1) * 100, 1)
        else:
            result[key] = None
    return result


def _build_strategy(close, stop_price, supports, fast_col, slow_col,
                    df, fast_ma, slow_ma):
    """回傳 (積極進場, 保守進場, [失效條件]) 三個字串"""
    latest         = df.iloc[-1]
    gap            = float(latest[fast_col]) - float(latest[slow_col])
    slow_ma_price  = float(latest[slow_col])
    fast_ma_price  = float(latest[fast_col])
    below_sup      = sorted([s for s in supports if s < close])
    above_sup      = sorted([s for s in supports if s > close])
    nearest_sup    = below_sup[-1] if below_sup else None
    nearest_res    = above_sup[0]  if above_sup else None

    if gap > 0:
        if nearest_sup and (close - nearest_sup) / close < 0.06:
            aggressive = (f"現價距支撐 {nearest_sup:.1f} 僅 "
                          f"{(close-nearest_sup)/close*100:.1f}%，"
                          f"可在此附近分批試單，止損設 {stop_price:.1f}")
        else:
            aggressive = (f"多頭排列中，可現價附近建倉，"
                          f"止損 {stop_price:.1f}")
        if nearest_res:
            conservative = (f"等突破壓力 {nearest_res:.1f} 後量縮"
                            f"回測 MA{fast_ma}（{fast_ma_price:.1f}）"
                            f"站穩再進，確認有效突破")
        else:
            conservative = (f"等量縮回測 MA{fast_ma}（{fast_ma_price:.1f}）"
                            f"不破再進，止損 {stop_price:.1f}")
    else:
        aggressive    = "目前空頭排列，建議暫緩佈局"
        conservative  = (f"等 MA{fast_ma} 站回 MA{slow_ma}"
                         f"（{slow_ma_price:.1f}）確認轉強後再考慮")

    fails = []
    if nearest_sup:
        fails.append(f"收盤跌破支撐 {nearest_sup:.1f} → 停損出場")
    fails.append(f"收盤跌破 MA{slow_ma}（{slow_ma_price:.1f}）→ 型態失效")
    if gap > 0:
        fails.append(f"MA{fast_ma} 重新跌破 MA{slow_ma} → 出場")

    return aggressive, conservative, fails


def _detect_divergence(df, lookback=25):
    """
    偵測 OBV / RSI 量價背離（頂背離 / 底背離）。
    把最近 lookback 根 K 棒分成前後兩半比較。
    """
    hits = []
    if len(df) < lookback + 5:
        return hits

    window = df.tail(lookback)
    half   = lookback // 2
    first  = window.iloc[:half]
    second = window.iloc[half:]

    tol_pct = 0.015  # 1.5% 才算有效差異

    # OBV 背離
    if "OBV" in df.columns:
        ph1, ph2 = first["Close"].max(),  second["Close"].max()
        oh1, oh2 = first["OBV"].max(),    second["OBV"].max()
        pl1, pl2 = first["Close"].min(),  second["Close"].min()
        ol1, ol2 = first["OBV"].min(),    second["OBV"].min()

        if all(pd.notna(x) for x in (ph1, ph2, oh1, oh2)):
            if ph2 > ph1 * (1 + tol_pct) and oh2 < oh1:
                hits.append("OBV頂背離（價漲量縮，假突破風險）")
        if all(pd.notna(x) for x in (pl1, pl2, ol1, ol2)):
            if pl2 < pl1 * (1 - tol_pct) and ol2 > ol1:
                hits.append("OBV底背離（量縮下跌，可能築底）")

    # RSI 背離
    if "RSI14" in df.columns:
        ph1, ph2 = first["Close"].max(),   second["Close"].max()
        rh1, rh2 = first["RSI14"].max(),   second["RSI14"].max()
        pl1, pl2 = first["Close"].min(),   second["Close"].min()
        rl1, rl2 = first["RSI14"].min(),   second["RSI14"].min()

        if all(pd.notna(x) for x in (ph1, ph2, rh1, rh2)):
            if ph2 > ph1 * (1 + tol_pct) and rh2 < rh1 - 3:
                hits.append("RSI頂背離（動能減弱，注意轉折）")
        if all(pd.notna(x) for x in (pl1, pl2, rl1, rl2)):
            if pl2 < pl1 * (1 - tol_pct) and rl2 > rl1 + 3:
                hits.append("RSI底背離（跌勢趨緩，反彈待確認）")

    return hits


# ── 回測引擎 ──────────────────────────────────────────────────────────────────

def run_backtest(df, fast_col, slow_col, fast_ma, slow_ma,
                 initial_capital=1_000_000,
                 use_atr_stop=False, atr_mult=2.0, stop_pct=7):
    """
    純 pandas 回測：均線交叉入場、死叉或止損出場。
    台股手續費：買入 0.0855%，賣出 0.3855%（手續費 + 0.3% 證交稅）。
    回傳 (stats_dict, trades_df, equity_series, bh_series)
    """
    BUY_FEE  = 0.000855
    SELL_FEE = 0.003855

    cash        = float(initial_capital)
    shares      = 0
    entry_price = 0.0
    entry_cost  = 0.0
    entry_date  = None
    stop_price  = 0.0
    trades = []
    eq_dates, eq_values = [], []

    for date, row in df.iterrows():
        close = float(row["Close"])
        sig   = int(row.get("signal", 0))

        # 止損出場（優先於訊號）
        if shares > 0 and close <= stop_price:
            proceeds = shares * close * (1 - SELL_FEE)
            pnl = proceeds - entry_cost
            cash += proceeds
            trades.append({
                "入場日": entry_date.strftime("%Y-%m-%d"),
                "出場日": date.strftime("%Y-%m-%d"),
                "入場價": round(entry_price, 2),
                "出場價": round(close, 2),
                "出場類型": "止損",
                "持有天數": (date - entry_date).days,
                "損益%": round(pnl / entry_cost * 100, 2) if entry_cost > 0 else 0.0,
                "損益(元)": round(pnl),
            })
            shares = 0

        # 訊號出場
        elif shares > 0 and sig == -1:
            proceeds = shares * close * (1 - SELL_FEE)
            pnl = proceeds - entry_cost
            cash += proceeds
            trades.append({
                "入場日": entry_date.strftime("%Y-%m-%d"),
                "出場日": date.strftime("%Y-%m-%d"),
                "入場價": round(entry_price, 2),
                "出場價": round(close, 2),
                "出場類型": "訊號",
                "持有天數": (date - entry_date).days,
                "損益%": round(pnl / entry_cost * 100, 2) if entry_cost > 0 else 0.0,
                "損益(元)": round(pnl),
            })
            shares = 0

        # 買入（無持倉才進場）
        if shares == 0 and sig == 1:
            cost_per = close * (1 + BUY_FEE)
            _shares  = int(cash * 0.95 / cost_per)
            if _shares > 0:
                shares      = _shares
                entry_cost  = shares * cost_per
                entry_price = close
                entry_date  = date
                cash       -= entry_cost
                if use_atr_stop and pd.notna(row.get("ATR14")):
                    stop_price = close - atr_mult * float(row["ATR14"])
                else:
                    stop_price = close * (1 - stop_pct / 100)

        eq_dates.append(date)
        eq_values.append(cash + shares * close)

    # 期末強制出場
    if shares > 0:
        close = float(df["Close"].iloc[-1])
        date  = df.index[-1]
        proceeds = shares * close * (1 - SELL_FEE)
        pnl = proceeds - entry_cost
        cash += proceeds
        trades.append({
            "入場日": entry_date.strftime("%Y-%m-%d"),
            "出場日": date.strftime("%Y-%m-%d"),
            "入場價": round(entry_price, 2),
            "出場價": round(close, 2),
            "出場類型": "期末",
            "持有天數": (date - entry_date).days,
            "損益%": round(pnl / entry_cost * 100, 2) if entry_cost > 0 else 0.0,
            "損益(元)": round(pnl),
        })
        eq_values[-1] = cash

    eq_s = pd.Series(eq_values, index=eq_dates, dtype=float)

    # 績效指標
    final_val  = float(eq_s.iloc[-1])
    total_ret  = (final_val / initial_capital - 1) * 100
    daily_ret  = eq_s.pct_change().dropna()
    sharpe     = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) \
                 if len(daily_ret) > 1 and daily_ret.std() > 0 else 0.0
    peak       = eq_s.cummax()
    max_dd     = float(((eq_s - peak) / peak).min() * 100)
    _cl = df["Close"].dropna()
    first_close = float(_cl.iloc[0]) if not _cl.empty else 1
    last_close  = float(_cl.iloc[-1]) if not _cl.empty else first_close
    bh_ret     = (last_close / first_close - 1) * 100
    bh_eq      = df["Close"] / first_close * initial_capital

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["入場日","出場日","入場價","出場價","出場類型","持有天數","損益%","損益(元)"]
    )
    if not trades_df.empty:
        wins   = trades_df[trades_df["損益%"] > 0]
        losses = trades_df[trades_df["損益%"] <= 0]
        wr     = len(wins) / len(trades_df) * 100
        avg_w  = float(wins["損益%"].mean())   if not wins.empty   else 0.0
        avg_l  = float(losses["損益%"].mean()) if not losses.empty else 0.0
        gp     = float(wins["損益(元)"].sum())         if not wins.empty   else 0.0
        gl     = float(abs(losses["損益(元)"].sum()))  if not losses.empty else 0.0
        pf     = round(gp / gl, 2) if gl > 0 else (99.0 if gp > 0 else 0.0)
    else:
        wr = avg_w = avg_l = pf = 0.0

    stats = {
        "初始資金":       initial_capital,
        "期末資金":       round(final_val),
        "總報酬%":        round(total_ret, 2),
        "買持報酬%":      round(bh_ret, 2),
        "超額報酬%":      round(total_ret - bh_ret, 2),
        "年化Sharpe":     round(sharpe, 2),
        "最大回撤%":      round(max_dd, 2),
        "交易次數":       len(trades_df),
        "勝率%":          round(wr, 1),
        "平均獲利%":      round(avg_w, 2),
        "平均虧損%":      round(avg_l, 2),
        "Profit Factor":  pf,
    }
    return stats, trades_df, eq_s, bh_eq


def run_pattern_backtest(df, fast_col, slow_col, fast_ma, slow_ma,
                         initial_capital=1_000_000,
                         stop_pct=7, use_atr_stop=False, atr_mult=2.0):
    """
    走步前進回測：型態突破策略。
    入場：型態成立 + 收盤在觸發價 -2%~+8% + MA快>MA慢 + ADX≥20 + 無頂背離
    出場：死叉 or 止損（交由 run_backtest 處理，邏輯與 MA 策略相同）
    注意：pivot 偵測使用 ±5 根視窗，最後 5 根 K 棒的 pivot 確認有輕微前瞻偏差，
          與實盤系統一致，屬可接受範圍。
    """
    n = len(df)
    sig = np.zeros(n, dtype=int)
    MIN_BARS = 65

    fc_arr  = df[fast_col].values.astype(float)
    sc_arr  = df[slow_col].values.astype(float)
    adx_arr = df["ADX14"].values.astype(float) if "ADX14" in df.columns else np.full(n, np.nan)

    for i in range(MIN_BARS, n):
        sub = df.iloc[:i + 1]
        c   = sub["Close"].values.astype(float)
        v   = sub["Volume"].values.astype(float)

        # 型態偵測
        pats = []
        for fn, needs_v in [(_detect_vcp,          True),
                             (_detect_flat_base,    True),
                             (_detect_double_bottom, False),
                             (_detect_cup_handle,   True)]:
            r = fn(c, v) if needs_v else fn(c)
            if r:
                pats.append(r)
        if not pats:
            continue

        trigger = _extract_trigger_price(pats)
        if trigger is None:
            continue

        close = c[-1]
        if not (trigger * 0.98 <= close <= trigger * 1.08):
            continue

        # MA 趨勢確認
        mf, ms = fc_arr[i], sc_arr[i]
        if np.isnan(mf) or np.isnan(ms) or mf <= ms:
            continue

        # ADX ≥ 20
        adx = adx_arr[i]
        if np.isnan(adx) or adx < 20:
            continue

        # 頂背離過濾
        if any("頂背離" in d for d in _detect_divergence(sub)):
            continue

        # 突破量能確認
        vol_avg20 = float(np.mean(v[-20:])) if len(v) >= 20 else 0
        if vol_avg20 > 0 and v[-1] < vol_avg20 * 1.5:
            continue

        # 收法確認：收陽線且收在當日振幅上段 ≥ 60%
        if "High" in sub.columns and "Low" in sub.columns and "Open" in sub.columns:
            _h = float(sub["High"].iloc[-1])
            _l = float(sub["Low"].iloc[-1])
            _o = float(sub["Open"].iloc[-1])
            _rng = _h - _l
            _pos_ok = (_rng <= 0) or ((close - _l) / _rng >= 0.6)
            if not (_pos_ok and close >= _o):
                continue

        sig[i] = 1

    # 死叉 → 出場訊號（與 MA 策略一致）
    for i in range(1, n):
        fpr, spr = fc_arr[i - 1], sc_arr[i - 1]
        fcr, scr = fc_arr[i],     sc_arr[i]
        if not any(np.isnan(x) for x in [fpr, spr, fcr, scr]):
            if fpr > spr and fcr <= scr:
                sig[i] = -1

    df2 = df.copy()
    df2["signal"] = sig
    return run_backtest(df2, fast_col, slow_col, fast_ma, slow_ma,
                        initial_capital=initial_capital,
                        use_atr_stop=use_atr_stop, atr_mult=atr_mult,
                        stop_pct=stop_pct)


@st.cache_data(ttl=1800)
def _detect_regime(df, n_states=3):
    """
    用 GaussianMixture 偵測股票自身的市場狀態：多頭 / 橫盤 / 空頭
    特徵：20日均報酬、20日波動率、成交量比
    不依賴 hmmlearn，sklearn 已內建。
    """
    try:
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler

        if len(df) < 60:
            return "資料不足"

        close  = df["Close"].values.astype(float)
        volume = df["Volume"].values.astype(float)

        log_ret = np.log(close[1:] / close[:-1])
        ret_20  = pd.Series(log_ret).rolling(20).mean().values
        vol_20  = pd.Series(log_ret).rolling(20).std().values * np.sqrt(252)
        vol_idx = volume[1:]
        vr      = vol_idx / (pd.Series(vol_idx).rolling(20).mean().values + 1e-9)

        X = np.column_stack([ret_20, vol_20, vr])
        mask = ~np.isnan(X).any(axis=1)
        X_clean = X[mask]

        if len(X_clean) < 30:
            return "資料不足"

        X_s    = StandardScaler().fit_transform(X_clean)
        states = GaussianMixture(n_components=n_states, random_state=42,
                                  max_iter=300).fit_predict(X_s)

        state_ret = {s: float(X_clean[states == s, 0].mean()) for s in range(n_states)}
        ordered   = sorted(state_ret, key=lambda s: state_ret[s])
        label_map = {ordered[0]: "空頭", ordered[-1]: "多頭"}
        if n_states == 3:
            label_map[ordered[1]] = "橫盤"

        return label_map.get(int(states[-1]), "未知")
    except Exception:
        return "無法判斷"


@st.cache_data(ttl=3600)
def _fetch_benchmark(period):
    """抓 0050.TW 作為大盤基準"""
    try:
        df = yf.download("0050.TW", period=period, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        return df if not df.empty else None
    except Exception:
        return None


@st.cache_data(ttl=86400)
def _get_fin_health(symbol):
    """從 yfinance 抓財務健康指標（快取 24 小時）"""
    try:
        info = yf.Ticker(symbol).info
        eps  = info.get("trailingEps")
        eg   = info.get("earningsGrowth")
        rg   = info.get("revenueGrowth")
        roe  = info.get("returnOnEquity")
        pe   = info.get("trailingPE")
        pb   = info.get("priceToBook")

        if eps is None:
            grade = "無資料"
        elif eps < 0:
            grade = "虧損"
        else:
            growing   = (eg is not None and eg > 0.05) or (rg is not None and rg > 0.05)
            declining = (eg is not None and eg < -0.2)  or (rg is not None and rg < -0.2)
            if growing:
                grade = "良好"
            elif declining:
                grade = "衰退"
            else:
                grade = "普通"

        return {
            "grade": grade,
            "eps":  round(eps, 2)        if eps is not None              else None,
            "eg":   round(eg  * 100, 1)  if eg  is not None              else None,
            "rg":   round(rg  * 100, 1)  if rg  is not None              else None,
            "roe":  round(roe * 100, 1)  if roe is not None              else None,
            "pe":   round(pe, 1)         if pe  is not None and pe  > 0  else None,
            "pb":   round(pb, 2)         if pb  is not None and pb  > 0  else None,
        }
    except Exception:
        return {"grade": "無資料", "eps": None, "eg": None,
                "rg": None, "roe": None, "pe": None, "pb": None}


def _calc_rs(df, bench_df, n=63):
    """個股相對強度 = 個股N日報酬 - 0050 N日報酬（正值＝跑贏大盤）"""
    try:
        if bench_df is None or len(df) <= n or len(bench_df) <= n:
            return None
        s_ret = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-1 - n]) - 1) * 100
        b_ret = (float(bench_df["Close"].iloc[-1]) / float(bench_df["Close"].iloc[-1 - n]) - 1) * 100
        return round(s_ret - b_ret, 1)
    except Exception:
        return None


# ── 型態偵測 ──────────────────────────────────────────────────────────────────

def _detect_vcp(c, v):
    """VCP：每次回檔幅度與量能逐次收縮"""
    n = len(c)
    if n < 40:
        return None
    w = 5
    ph, pl = [], []
    for i in range(w, n - w):
        seg = c[i-w:i+w+1]
        if c[i] >= max(seg) - 1e-9:
            ph.append(i)
        elif c[i] <= min(seg) + 1e-9:
            pl.append(i)
    if len(ph) < 2 or len(pl) < 2:
        return None
    corrs = []
    for hi in ph:
        nxt = [li for li in pl if li > hi]
        if not nxt:
            continue
        li = nxt[0]
        pct = (c[hi] - c[li]) / c[hi] * 100
        avg_v = float(np.mean(v[max(0, li-3):li+4]))
        corrs.append((pct, avg_v, float(c[hi])))
    if len(corrs) < 3:
        return None
    recent = corrs[-4:]
    shrink = all(recent[i][0] > recent[i+1][0] for i in range(len(recent)-1))
    if not shrink:
        return None
    vol_ok = all(recent[i][1] >= recent[i+1][1] for i in range(len(recent)-1))
    strength = "強" if (vol_ok and len(recent) >= 3) else "中"
    last_hi = recent[-1][2]
    return f"VCP {len(recent)}次收縮（{strength}），觸發 {last_hi:.1f}"


def _detect_double_bottom(c):
    """雙底（W底）：兩個相近低點 + 頸線確認"""
    n = len(c)
    if n < 40:
        return None
    w = 7
    raw_lows = []
    for i in range(w, n - w):
        if c[i] <= min(c[i-w:i+w+1]) + 1e-9:
            raw_lows.append(i)
    merged = []
    for li in raw_lows:
        if not merged or li - merged[-1] > w * 2:
            merged.append(li)
    if len(merged) < 2:
        return None
    for i in range(len(merged) - 1):
        l1, l2 = merged[i], merged[i+1]
        if l2 - l1 < 15:
            continue
        p1, p2 = c[l1], c[l2]
        if abs(p1 - p2) / min(p1, p2) > 0.05:
            continue
        neck = max(c[l1:l2+1])
        if neck < min(p1, p2) * 1.05:
            continue
        dist = (neck - c[-1]) / neck * 100
        if -10 <= dist <= 5:
            return f"雙底，頸線 {neck:.1f}（距 {dist:.1f}%）"
    return None


def _detect_flat_base(c, v):
    """平台底：近期低波動橫盤整理，量縮"""
    n = len(c)
    for period in [25, 35, 50]:
        if n < period + 20:
            continue
        seg, sv = c[-period:], v[-period:]
        h, lo = max(seg), min(seg)
        rng = (h - lo) / h * 100
        if rng > 12:
            continue
        if h < max(c[-min(120, n):]) * 0.75:
            continue
        mid = period // 2
        vol_ratio = np.mean(sv[mid:]) / (np.mean(sv[:mid]) + 1e-9)
        note = "量縮" if vol_ratio < 0.85 else ""
        return f"平台底 {period}日，波動 {rng:.1f}%{(' '+note) if note else ''}，觸發 {h:.1f}"
    return None


def _detect_cup_handle(c, v):
    """杯柄型態：圓弧底 + 小幅回調柄"""
    n = len(c)
    if n < 60:
        return None
    lb = min(120, n)
    seg = c[-lb:]
    t1, t2 = lb // 3, 2 * lb // 3
    lh = max(seg[:t1])
    lhi = int(np.argmax(seg[:t1]))
    cup_lo = min(seg[lhi:t2])
    cup_li = lhi + int(np.argmin(seg[lhi:t2]))
    depth = (lh - cup_lo) / lh * 100
    if depth < 10 or depth > 40:
        return None
    r_seg = seg[cup_li:t2+1]
    if len(r_seg) < 5:
        return None
    rh = max(r_seg)
    rhi = cup_li + int(np.argmax(r_seg))
    if abs(rh - lh) / lh > 0.08:
        return None
    handle = seg[rhi:]
    if len(handle) < 5:
        return None
    hlo = min(handle)
    hdepth = (rh - hlo) / rh * 100
    if hdepth < 3 or hdepth > 20:
        return None
    if hlo < cup_lo + (lh - cup_lo) * 0.5:
        return None
    dist = (rh - seg[-1]) / rh * 100
    return f"杯柄，杯深 {depth:.1f}%，突破點 {rh:.1f}（距 {dist:.1f}%）"


def detect_all_patterns(df):
    c = df["Close"].values.astype(float)
    v = df["Volume"].values.astype(float)
    hits = []
    for fn in (_detect_vcp, _detect_flat_base, _detect_double_bottom, _detect_cup_handle):
        if fn in (_detect_vcp, _detect_flat_base):
            r = fn(c, v)
        else:
            r = fn(c) if fn == _detect_double_bottom else fn(c, v)
        if r:
            hits.append(r)
    hits.extend(_detect_divergence(df))
    return hits


def _draw_trendlines(fig, df):
    """
    自動畫有效趨勢線：掃所有樞紐點組合，找觸碰點最多的線。
    至少需要 3 個觸碰點才畫（標準技術分析：2 點只是試驗線，3 點才確認）。
    觸碰點用小圓點標記，標籤顯示 ×N。
    """
    high  = df["High"].values.astype(float)
    low   = df["Low"].values.astype(float)
    dates = df.index
    n = len(dates)
    w = max(5, n // 25)

    def _pivots(arr, is_high):
        pts = []
        for i in range(w, n - w):
            seg = arr[i-w:i+w+1]
            cond = arr[i] >= max(seg) - 1e-9 if is_high else arr[i] <= min(seg) + 1e-9
            if cond:
                pts.append((i, float(arr[i])))
        merged = []
        for p in pts:
            if not merged or p[0] - merged[-1][0] > w:
                merged.append(p)
        return merged

    def _best_line(pivots, tol=0.018):
        """找觸碰點最多的趨勢線，回傳 (p_start, slope, touch_list) 或 None"""
        if len(pivots) < 2:
            return None
        best_result = None
        best_cnt = 0
        for i in range(len(pivots)):
            for j in range(i + 1, len(pivots)):
                x1, y1 = pivots[i]
                x2, y2 = pivots[j]
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                touches = []
                for xi, yi in pivots:
                    y_line = y1 + slope * (xi - x1)
                    if abs(yi - y_line) / max(y_line, 1e-9) < tol:
                        touches.append((xi, yi))
                if len(touches) > best_cnt:
                    best_cnt = len(touches)
                    best_result = (pivots[i], slope, touches)
        return best_result if best_cnt >= 3 else None

    def _render(p_start, slope, touches, color, label):
        x0, y0 = p_start
        x_first = touches[0][0]
        y_first = y0 + slope * (x_first - x0)
        y_end   = y0 + slope * (n - 1 - x0)
        lw = 1.5 + 0.4 * min(len(touches) - 2, 4)  # 觸碰越多線越粗

        fig.add_shape(
            type="line",
            x0=dates[x_first], x1=dates[-1],
            y0=y_first, y1=y_end,
            line=dict(color=color, width=lw),
            row=1, col=1
        )
        # 觸碰點小圓
        fig.add_trace(go.Scatter(
            x=[dates[xi] for xi, _ in touches],
            y=[yi for _, yi in touches],
            mode="markers",
            marker=dict(symbol="circle-open", size=9, color=color,
                        line=dict(width=2, color=color)),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=1)
        fig.add_annotation(
            x=dates[-1], y=y_end,
            text=f"  {label} ×{len(touches)}",
            font=dict(color=color, size=10),
            xanchor="left", showarrow=False,
            xref="x", yref="y"
        )

    highs = _pivots(high, True)
    lows  = _pivots(low,  False)

    res_lo = _best_line(lows)
    if res_lo:
        p, slope, touches = res_lo
        color = "rgba(38,166,154,0.9)" if slope >= 0 else "rgba(220,80,80,0.65)"
        label = "上升支撐" if slope >= 0 else "下降支撐"
        _render(p, slope, touches, color, label)

    res_hi = _best_line(highs)
    if res_hi:
        p, slope, touches = res_hi
        color = "rgba(239,83,80,0.9)" if slope <= 0 else "rgba(255,152,0,0.9)"
        label = "下降壓力" if slope <= 0 else "上升壓力"
        _render(p, slope, touches, color, label)


def build_chart(df, symbol, fast_col, slow_col, fast_ma, slow_ma, supports, stop_pct,
                show_bb=True, use_atr_stop=False, atr_mult=2.0, momentum_ind="RSI"):
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.57, 0.17, 0.26],
        subplot_titles=[f"{symbol} 技術走勢", "成交量", momentum_ind]
    )

    # K棒
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="K棒",
        increasing_line_color="#ef5350", increasing_fillcolor="#ef5350",
        decreasing_line_color="#26a69a", decreasing_fillcolor="#26a69a",
    ), row=1, col=1)

    # 均線
    MA_STYLE = {5: ("#FF9800", 1.2), 10: ("#42A5F5", 1.2), 20: ("#CE93D8", 1.8), 60: ("#EF5350", 2.2)}
    for ma, (color, width) in MA_STYLE.items():
        fig.add_trace(go.Scatter(
            x=df.index, y=df[f"MA{ma}"],
            name=f"MA{ma}", line=dict(color=color, width=width)
        ), row=1, col=1)

    # Bollinger Bands
    if show_bb:
        close_arr = df["Close"].values.astype(float)
        bb_mid, bb_up, bb_lo = _calc_bb(close_arr)
        fig.add_trace(go.Scatter(
            x=df.index, y=bb_up, name="BB上軌",
            line=dict(color="rgba(150,150,255,0.55)", width=1, dash="dot"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=bb_lo, name="BB下軌",
            line=dict(color="rgba(150,150,255,0.55)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(150,150,255,0.05)",
        ), row=1, col=1)

    buy_df  = df[df["signal"] ==  1]
    sell_df = df[df["signal"] == -1]

    if not buy_df.empty:
        fig.add_trace(go.Scatter(
            x=buy_df.index, y=buy_df["Low"] * 0.983,
            mode="markers", name=f"入場 MA{fast_ma}↑MA{slow_ma}",
            marker=dict(symbol="triangle-up", size=14, color="#00E676",
                        line=dict(color="white", width=1))
        ), row=1, col=1)

    if not sell_df.empty:
        fig.add_trace(go.Scatter(
            x=sell_df.index, y=sell_df["High"] * 1.017,
            mode="markers", name=f"出場 MA{fast_ma}↓MA{slow_ma}",
            marker=dict(symbol="triangle-down", size=14, color="#FF1744",
                        line=dict(color="white", width=1))
        ), row=1, col=1)

    # 止損線
    buy_dates  = df[df["signal"] ==  1].index.tolist()
    sell_dates = df[df["signal"] == -1].index.tolist()
    stop_label = f"ATR×{atr_mult} 止損" if use_atr_stop else f"止損 -{stop_pct}%"
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines", name=stop_label,
        line=dict(color="rgba(255,152,0,0.8)", width=1.5, dash="dot")
    ), row=1, col=1)

    sl_label_x, sl_label_y, sl_label_text = [], [], []
    for entry_date in buy_dates:
        entry_price = float(df.loc[entry_date, "Close"])
        if use_atr_stop and "ATR14" in df.columns:
            atr_val    = float(df.loc[entry_date, "ATR14"])
            stop_price = round(entry_price - atr_mult * atr_val, 2)
        else:
            stop_price = round(entry_price * (1 - stop_pct / 100), 2)
        future_sells = [s for s in sell_dates if s > entry_date]
        exit_date    = future_sells[0] if future_sells else df.index[-1]
        fig.add_shape(type="line",
            x0=entry_date, x1=exit_date, y0=stop_price, y1=stop_price,
            line=dict(color="rgba(255,152,0,0.75)", width=1.5, dash="dot"), row=1, col=1)
        sl_label_x.append(entry_date)
        sl_label_y.append(stop_price)
        sl_label_text.append(f"{stop_price:.1f}")

    if sl_label_x:
        fig.add_trace(go.Scatter(
            x=sl_label_x, y=sl_label_y, mode="text", text=sl_label_text,
            textfont=dict(color="#FFB74D", size=9), textposition="bottom center",
            showlegend=False, hoverinfo="skip"
        ), row=1, col=1)

    # 支撐線
    for lvl in supports:
        fig.add_shape(type="line",
            x0=df.index[0], x1=df.index[-1], y0=lvl, y1=lvl,
            line=dict(color="rgba(255,235,59,0.55)", width=1.3, dash="dash"), row=1, col=1)
        fig.add_annotation(
            x=df.index[-1], y=lvl, text=f"  支撐 {lvl:.1f}",
            font=dict(color="#FFF176", size=11),
            xanchor="left", showarrow=False, xref="x", yref="y")

    # 52週高低水平線
    high_52 = float(df["High"].tail(252).max())
    low_52  = float(df["Low"].tail(252).min())
    for val, label, color in [
        (high_52, "52週高", "rgba(255,100,100,0.65)"),
        (low_52,  "52週低", "rgba(38,220,154,0.65)"),
    ]:
        fig.add_shape(type="line",
            x0=df.index[0], x1=df.index[-1], y0=val, y1=val,
            line=dict(color=color, width=1, dash="longdash"), row=1, col=1)
        fig.add_annotation(
            x=df.index[-1], y=val, text=f"  {label} {val:.1f}",
            font=dict(color=color, size=10),
            xanchor="left", showarrow=False, xref="x", yref="y")

    # 成交量 + 20日量均
    vol_colors = ["#ef5350" if float(c) >= float(o) else "#26a69a"
                  for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"],
                         marker_color=vol_colors, showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Volume"].rolling(20).mean(),
        name="Vol MA20", line=dict(color="rgba(255,255,100,0.6)", width=1),
        showlegend=False
    ), row=2, col=1)

    # ── 動能指標子圖 ──────────────────────────────────────────────────────────
    close_arr = df["Close"].values.astype(float)
    high_arr  = df["High"].values.astype(float)
    low_arr   = df["Low"].values.astype(float)

    if momentum_ind == "RSI":
        rsi_vals = df["RSI14"].values if "RSI14" in df.columns else None
        if rsi_vals is not None:
            fig.add_trace(go.Scatter(
                x=df.index, y=rsi_vals, name="RSI(14)",
                line=dict(color="#FFD54F", width=1.5)
            ), row=3, col=1)
            for lvl, clr in [(70, "rgba(239,83,80,0.55)"),
                             (50, "rgba(200,200,200,0.3)"),
                             (30, "rgba(38,166,154,0.55)")]:
                fig.add_shape(type="line",
                    x0=df.index[0], x1=df.index[-1], y0=lvl, y1=lvl,
                    line=dict(color=clr, width=1, dash="dot"), row=3, col=1)
            fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.07)",
                          line_width=0, row=3, col=1)
            fig.add_hrect(y0=0, y1=30, fillcolor="rgba(38,166,154,0.07)",
                          line_width=0, row=3, col=1)
            fig.update_yaxes(range=[0, 100], row=3, col=1)

    elif momentum_ind == "KD":
        k_vals, d_vals = _calc_kd(high_arr, low_arr, close_arr)
        fig.add_trace(go.Scatter(x=df.index, y=k_vals, name="K(9)",
                                  line=dict(color="#FFD54F", width=1.5)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=d_vals, name="D(3)",
                                  line=dict(color="#EF5350", width=1.5)), row=3, col=1)
        for lvl, clr in [(80, "rgba(239,83,80,0.55)"),
                         (50, "rgba(200,200,200,0.3)"),
                         (20, "rgba(38,166,154,0.55)")]:
            fig.add_shape(type="line",
                x0=df.index[0], x1=df.index[-1], y0=lvl, y1=lvl,
                line=dict(color=clr, width=1, dash="dot"), row=3, col=1)
        fig.add_hrect(y0=80, y1=100, fillcolor="rgba(239,83,80,0.07)",
                      line_width=0, row=3, col=1)
        fig.add_hrect(y0=0, y1=20, fillcolor="rgba(38,166,154,0.07)",
                      line_width=0, row=3, col=1)

    elif momentum_ind == "OBV":
        if "OBV" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["OBV"], name="OBV",
                                      line=dict(color="#FFD54F", width=1.5)), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["OBV_MA20"], name="OBV MA20",
                                      line=dict(color="#EF5350", width=1, dash="dot")), row=3, col=1)
            fig.add_shape(type="line",
                x0=df.index[0], x1=df.index[-1], y0=0, y1=0,
                line=dict(color="rgba(255,255,255,0.3)", width=1), row=3, col=1)

    else:  # MACD
        macd_vals, sig_vals, hist_vals = _calc_macd(close_arr)
        hist_colors = ["#ef5350" if v >= 0 else "#26a69a" for v in hist_vals]
        fig.add_trace(go.Bar(x=df.index, y=hist_vals, name="MACD柱",
                             marker_color=hist_colors, showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=macd_vals, name="MACD",
                                  line=dict(color="#FFD54F", width=1.5)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=sig_vals, name="Signal",
                                  line=dict(color="#EF5350", width=1.5)), row=3, col=1)
        fig.add_shape(type="line",
            x0=df.index[0], x1=df.index[-1], y0=0, y1=0,
            line=dict(color="rgba(255,255,255,0.3)", width=1), row=3, col=1)

    fig.update_layout(
        height=820,
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=95, t=40, b=10)
    )
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

    _draw_trendlines(fig, df)

    return fig


# ── 模擬交易 helpers ──────────────────────────────────────────────────────────
_BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
PAPER_FILE          = os.path.join(_BASE_DIR, "paper_trades.json")
SCAN_HISTORY_FILE   = os.path.join(_BASE_DIR, "scan_history.json")
SHEET_ID            = "17o-c7bQXcTk53DwRfQGESnCyxjA2owY8D7-NJzurTxQ"
EMPTY_TRADES        = {"positions": [], "closed": []}

def _load_scan_history():
    try:
        with open(SCAN_HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_scan_history(today_str, codes):
    hist = _load_scan_history()
    hist[today_str] = list(codes)
    keep = sorted(hist.keys())[-10:]
    hist = {d: hist[d] for d in keep}
    try:
        with open(SCAN_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(hist, f, ensure_ascii=False)
    except Exception:
        pass

def _count_streak(code, history, today_str):
    count = 0
    for d in sorted(history.keys(), reverse=True):
        if d >= today_str:
            continue
        if code in history.get(d, []):
            count += 1
        else:
            break
    return count

@st.cache_resource
def _get_gsheet():
    """回傳 gspread worksheet，失敗回傳 None（fallback 到本機 JSON）"""
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        creds_info = dict(st.secrets["gcp_service_account"])
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
        gc = gspread.authorize(creds)
        return gc.open_by_key(SHEET_ID).sheet1
    except Exception:
        return None

def _load_trades():
    ws = _get_gsheet()
    if ws is not None:
        try:
            val = ws.acell("A1").value
            return json.loads(val) if val else EMPTY_TRADES
        except Exception:
            pass
    if not os.path.exists(PAPER_FILE):
        return EMPTY_TRADES
    try:
        with open(PAPER_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return EMPTY_TRADES

def _save_trades(data):
    ws = _get_gsheet()
    if ws is not None:
        try:
            ws.update_acell("A1", json.dumps(data, ensure_ascii=False))
            return
        except Exception:
            pass
    try:
        with open(PAPER_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

@st.cache_data(ttl=300)
def _fetch_latest_price(symbol):
    try:
        df = yf.download(symbol, period="5d", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return float(df["Close"].iloc[-1]) if not df.empty else None
    except Exception:
        return None


# ── 掃描清單預設 ──────────────────────────────────────────────────────────────
DEFAULT_WATCHLIST = """2330, 2454, 2317, 2382, 2308, 3711, 2303, 2379, 3034, 6415
2881, 2882, 2886, 2891, 2892, 5871, 2884, 2885, 2887, 2888
2412, 3045, 4904, 2002, 1301, 1303, 1326, 2207, 2408, 2395
4938, 3231, 2356, 2301, 2376, 2353, 3661, 6669, 2615, 2603
0050, 0056, 00878, 00881, 006208"""

# ── 抓全市場股票代號 ──────────────────────────────────────────────────────────
@st.cache_data(ttl=43200)
def _parse_isin_page(mode):
    """從 TWSE ISIN 頁面解析股票代號與產業別，回傳 (codes, {code: sector})"""
    import re
    url = f"https://isin.twse.com.tw/isin/C_public.jsp?strMode={mode}"
    r = requests.get(url, timeout=20, verify=False)
    html = r.content.decode("cp950", errors="ignore")
    rows = re.findall(
        r"<td bgcolor=#[A-F0-9]+>(\d+)　[^<]*</td>"
        r"(?:<td[^>]*>[^<]*</td>){3}"
        r"<td bgcolor=#[A-F0-9]+>([^<]*)</td>",
        html
    )
    exclude = ["權", "轉換", "存託", "特別", "債"]
    result, smap = [], {}
    for code, sec_type in rows:
        if any(x in sec_type for x in exclude):
            continue
        if (len(code) == 4 and code[0] != "0") or code.startswith("00"):
            result.append(code)
            smap[code] = sec_type.strip()
    return sorted(set(result)), smap

@st.cache_data(ttl=43200)
def fetch_twse_codes():
    try:
        import twstock
        codes = [
            code for code, info in twstock.codes.items()
            if getattr(info, 'market', '') == '上市'
            and ((len(code) == 4 and code[0] != '0') or code.startswith('00'))
        ]
        if codes:
            return sorted(set(codes))
    except Exception:
        pass
    try:
        codes, _ = _parse_isin_page(2)
        return codes
    except Exception:
        return []

@st.cache_data(ttl=43200)
def fetch_tpex_codes():
    try:
        import twstock
        codes = [
            code for code, info in twstock.codes.items()
            if getattr(info, 'market', '') == '上櫃'
            and ((len(code) == 4 and code[0] != '0') or code.startswith('00'))
        ]
        if codes:
            return sorted(set(codes))
    except Exception:
        pass
    try:
        codes, _ = _parse_isin_page(4)
        return codes
    except Exception:
        return []

@st.cache_data(ttl=43200)
def fetch_sector_map(suffix):
    """根據 suffix 回傳 {代號: 產業別} 對照表，優先用 twstock.group"""
    try:
        import twstock
        market = '上市' if suffix == '.TW' else '上櫃'
        smap = {
            code: getattr(info, 'group', '—')
            for code, info in twstock.codes.items()
            if getattr(info, 'market', '') == market and getattr(info, 'group', '')
        }
        if smap:
            return smap
    except Exception:
        pass
    mode = 2 if suffix == ".TW" else 4
    try:
        _, smap = _parse_isin_page(mode)
        return smap
    except Exception:
        return {}

@st.cache_data(ttl=3600)
def fetch_monthly_revenue():
    """抓取最新月營收 YoY，回傳 {代號: yoy_pct}"""
    from datetime import timedelta
    result = {}
    try:
        import urllib3; urllib3.disable_warnings()
        r = requests.get("https://openapi.twse.com.tw/v1/opendata/t187ap05_L",
                         timeout=25, verify=False,
                         headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200:
            for row in r.json():
                vals = list(row.values())
                if len(vals) >= 10:
                    try:
                        result[str(vals[2]).strip()] = float(vals[9])
                    except Exception:
                        pass
    except Exception:
        pass
    try:
        import urllib3; urllib3.disable_warnings()
        from datetime import datetime as _dt
        today = _dt.today()
        ref   = (today.replace(day=1) - timedelta(days=1)) if today.day >= 12 \
                else (today.replace(day=1) - timedelta(days=32))
        roc_year, month = ref.year - 1911, ref.month
        url  = "https://mops.twse.com.tw/mops/web/ajax_t05st10_ifrs"
        hdrs = {"User-Agent": "Mozilla/5.0",
                "Referer": "https://mops.twse.com.tw/mops/web/t05st10",
                "Content-Type": "application/x-www-form-urlencoded"}
        payload = {"encodeURIComponent": "1", "step": "1", "firstin": "1",
                   "off": "1", "TYPEK": "otc", "isnew": "false",
                   "year": str(roc_year), "month": f"{month:02d}"}
        r2   = requests.post(url, data=payload, headers=hdrs, timeout=25, verify=False)
        html = r2.content.decode("utf-8", errors="ignore")
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL)
        for row in rows:
            cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
            cells = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
            if len(cells) >= 10 and len(cells[0]) == 4 and cells[0].isdigit():
                try:
                    result[cells[0]] = float(cells[9].replace(",", ""))
                except Exception:
                    pass
    except Exception:
        pass
    return result


# ── 批次下載（100支一批，速度遠快於逐支下載）────────────────────────────────────
@st.cache_data(ttl=1800)
def fetch_multi(symbols_tuple, period):
    import concurrent.futures
    syms = list(symbols_tuple)
    def _dl():
        return yf.download(syms, period=period, auto_adjust=True,
                           progress=False, group_by="ticker", threads=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_dl)
        try:
            return fut.result(timeout=60)
        except concurrent.futures.TimeoutError:
            return pd.DataFrame()

def _extract_one(multi_df, symbol, n_syms):
    if not isinstance(multi_df.columns, pd.MultiIndex):
        return multi_df
    try:
        lvl0 = multi_df.columns.get_level_values(0)
        if symbol in lvl0:
            s = multi_df[symbol].dropna(how="all")
            return s if not s.empty else None
        return None
    except Exception:
        return None

def _extract_trigger_price(patterns):
    for p in patterns:
        m = re.search(r'頸線\s*([\d.]+)', p)
        if m:
            return float(m.group(1))
        m = re.search(r'突破點\s*([\d.]+)', p)
        if m:
            return float(m.group(1))
        m = re.search(r'觸發\s*([\d.]+)', p)
        if m:
            return float(m.group(1))
    return None


# ── 分析單支股票並回傳結果列 ──────────────────────────────────────────────────
def _analyze_one(code, stock_df, fast_ma, slow_ma, stop_pct, bench_df=None, sector="—", use_atr_stop=False, atr_mult=2.0, revenue_yoy=None):
    stock_df = stock_df.copy()
    stock_df.index = pd.to_datetime(stock_df.index)
    stock_df = stock_df.dropna(subset=["Close"])
    if len(stock_df) < max(slow_ma + 10, 40):
        return None

    df, fast_col, slow_col, supports = calc_indicators(stock_df, fast_ma, slow_ma, 7, 120)

    # 成交量過濾：20日均量 < 500張 或 均金額 < 500萬 → 跳過
    avg_vol = df["Volume"].tail(20).mean() if "Volume" in df.columns else 0
    latest_close = float(df["Close"].iloc[-1]) if not df.empty else 0
    if avg_vol < 500 or (avg_vol * latest_close) < 5_000_000:
        return None

    latest  = df.iloc[-1]
    close   = float(latest["Close"])
    gap_now = float(latest[fast_col]) - float(latest[slow_col])
    trend   = "多頭" if gap_now > 0 else "空頭"

    sig_df = df[df["signal"] != 0]
    if sig_df.empty:
        last_sig, last_date, days_ago = "無", "—", 999
    else:
        last_row  = sig_df.iloc[-1]
        last_sig  = "入場" if last_row["signal"] == 1 else "出場"
        last_date = last_row.name.strftime("%Y-%m-%d")
        days_ago  = (df.index[-1] - last_row.name).days

    if last_sig == "入場" and days_ago <= 5:
        tag = "近期黃金交叉"
    elif trend == "多頭" and last_sig == "入場":
        tag = "多頭持續中"
    elif last_sig == "出場" and days_ago <= 5:
        tag = "近期死亡交叉"
    elif trend == "空頭":
        tag = "空頭排列"
    else:
        tag = "觀察中"

    patterns   = detect_all_patterns(df)
    top_divs   = [p for p in patterns if "頂背離" in p]
    consol_pat = [p for p in patterns if any(k in p for k in ["VCP", "平台底", "杯柄", "雙底"])]

    # 過度延伸：1M > 25% 且無整理型態 → 追高風險（閾值與 calc_score 一致）
    _r1m = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-21]) - 1) * 100 \
           if len(df) > 21 else 0
    _r3m = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-63]) - 1) * 100 \
           if len(df) > 63 else 0
    is_extended = (_r1m > 25 and not consol_pat) or (_r3m > 60 and not consol_pat)

    # 型態突破偵測：所有型態現在都有觸發價，進場窗口 -2% ~ +8%
    _trigger = _extract_trigger_price(patterns)
    broke_out = (_trigger is not None and _trigger * 0.98 <= close <= _trigger * 1.08)

    if top_divs:
        action = "不看"
    elif is_extended:
        action = "不看"
    elif consol_pat and broke_out and trend == "多頭":
        action = "可買"        # 型態突破為主訊號
    elif consol_pat and trend == "多頭":
        action = "等待進場"    # 型態成形，等突破
    elif trend == "多頭" and tag in ("近期黃金交叉", "多頭持續中"):
        action = "等待進場"    # 純均線訊號，無型態
    else:
        action = "不看"

    if use_atr_stop and "ATR14" in df.columns and pd.notna(df["ATR14"].iloc[-1]):
        atr_now  = float(df["ATR14"].iloc[-1])
        _atr_stop = round(close - atr_mult * atr_now, 2)
        stop_val  = _atr_stop if 0 < _atr_stop < close else round(close * (1 - stop_pct / 100), 2)
    else:
        stop_val  = round(close * (1 - stop_pct / 100), 2)
    rs_val     = _calc_rs(df, bench_df) if bench_df is not None else None
    resist     = [s for s in (supports or []) if s > close]
    target_val = round(min(resist), 2) if resist else round(close * 1.15, 2)
    rr_val     = round((target_val - close) / (close - stop_val), 2) if 0 < stop_val < close else None

    # 方案B過濾：相對強度過弱或RR不足 → 可買降為等待進場
    if action == "可買":
        if rs_val is not None and rs_val < -15:
            action = "等待進場"
        elif rr_val is not None and rr_val < 1.5:
            action = "等待進場"

    # 收盤 > MA5 過濾：黃金交叉後若收盤已跌回 MA5 以下 → 等待進場
    if action == "可買":
        _ma5 = float(df["MA5"].iloc[-1]) if "MA5" in df.columns and pd.notna(df["MA5"].iloc[-1]) else 0
        if _ma5 > 0 and close < _ma5:
            action = "等待進場"

    # ADX 趨勢強度過濾：ADX < 20 代表盤整，均線交叉不可信
    if action == "可買":
        _adx = float(df["ADX14"].iloc[-1]) if "ADX14" in df.columns and pd.notna(df["ADX14"].iloc[-1]) else 0
        if _adx < 20:
            action = "等待進場"

    # 突破量能確認：量縮突破勝率低，需突破日成交量 ≥ 20 日均量 × 1.5
    if action == "可買":
        _vol_avg20 = float(df["Volume"].tail(20).mean()) if "Volume" in df.columns else 0
        _vol_today = float(latest["Volume"]) if pd.notna(latest.get("Volume", np.nan)) else 0
        if _vol_avg20 > 0 and _vol_today < _vol_avg20 * 1.5:
            action = "等待進場"

    # 收法確認：收陽線且收盤在當日振幅上段（≥ 60% 位置），排除上影線假突破
    if action == "可買":
        _h = float(latest["High"])  if pd.notna(latest.get("High",  np.nan)) else close
        _l = float(latest["Low"])   if pd.notna(latest.get("Low",   np.nan)) else close
        _o = float(latest["Open"])  if pd.notna(latest.get("Open",  np.nan)) else close
        _rng = _h - _l
        _pos_ok = (_rng <= 0) or ((close - _l) / _rng >= 0.6)
        if not (_pos_ok and close >= _o):
            action = "等待進場"

    # 多訊號確認：黃金交叉需搭配至少 1 個輔助訊號才升為可買
    if action == "可買":
        _vol_avg = df["Volume"].tail(20).mean()
        _vol_ratio = float(latest["Volume"]) / float(_vol_avg) if pd.notna(_vol_avg) and float(_vol_avg) > 0 else 0
        _rsi_now = float(df["RSI14"].iloc[-1]) if "RSI14" in df.columns and pd.notna(df["RSI14"].iloc[-1]) else 50
        _rsi_5d_min = float(df["RSI14"].tail(5).min()) if "RSI14" in df.columns else 50
        _aux = 0
        if _vol_ratio >= 1.5:                              _aux += 1  # 量能爆發
        if consol_pat:                                     _aux += 1  # 整理型態確認
        if _rsi_5d_min <= 40 and _rsi_now > 40:           _aux += 1  # RSI 從超賣回升
        if supports:
            _sup_dist = min(((close - s) / s for s in supports if s < close), default=1.0)
            if _sup_dist <= 0.03:                          _aux += 1  # 支撐反彈 3% 以內
        if _aux < 1:
            action = "等待進場"

    # 景氣循環產業：無月營收成長支撐時，可買降為等待進場
    _CYCLICAL = {"建材營造業", "水泥工業", "汽車工業"}
    if action == "可買" and sector in _CYCLICAL:
        if revenue_yoy is None or revenue_yoy <= 0:
            action = "等待進場"

    # 細分「等待進場」語意
    # broke_out=True 代表在觸發窗口內但某條件未達標（被從可買降下來）
    # broke_out=False 且 close > trigger*1.08 代表已超出窗口（已錯過）
    # 其他 consol_pat 情況 = 型態成形、尚未觸發（等突破）
    if action == "等待進場" and consol_pat:
        if broke_out:
            action = "條件未足"
        elif _trigger is not None and close > _trigger * 1.08:
            action = "已錯過"
        else:
            action = "等突破"

    score    = calc_score(df, patterns, fast_col, slow_col, rs_val, revenue_yoy)

    high_52  = df["High"].tail(252).max() if len(df) >= 150 else df["High"].max()
    dist_52h = round((close / high_52 - 1) * 100, 1) if high_52 > 0 else None
    rsi_now  = round(float(df["RSI14"].iloc[-1]), 1) if "RSI14" in df.columns and pd.notna(df["RSI14"].iloc[-1]) else None
    mom      = _calc_momentum(df)
    trigger  = _extract_trigger_price(patterns)

    def _fmt_mom(key):
        v = mom.get(key)
        return f"{v:+.1f}%" if v is not None else "—"

    return {
        "分數": score,
        "操作": action,
        "代號": code,
        "產業": sector,
        "收盤價": round(close, 2),
        "止損價": stop_val,
        "觸發價": f"{trigger:.1f}" if trigger is not None else "—",
        "RSI": rsi_now if rsi_now is not None else "—",
        "RS vs大盤": f"{rs_val:+.1f}%" if (rs_val is not None and not np.isnan(float(rs_val))) else "—",
        "1M%": _fmt_mom("R1M"),
        "3M%": _fmt_mom("R3M"),
        "距52週高%": f"{dist_52h:+.1f}%" if dist_52h is not None else "—",
        "RR": f"1:{rr_val:.1f}" if rr_val is not None else "—",
        "偵測型態": "、".join(patterns) if patterns else "—",
        "狀態": tag,
        "最近訊號": last_sig,
        "訊號日期": last_date,
        "幾天前": days_ago if days_ago < 999 else "—",
        "排列": trend,
    }

# ── 掃描主函式（批次下載版）────────────────────────────────────────────────────
def scan_stocks(codes, suffix, period, fast_ma, slow_ma, stop_pct, pb, bench_df=None, sector_map=None, use_atr_stop=False, atr_mult=2.0, revenue_dict=None):
    import time
    results = []
    batch_size = 50 if suffix == ".TWO" else 100
    total = len(codes)

    for i in range(0, total, batch_size):
        batch_codes = codes[i: i + batch_size]
        batch_syms  = [f"{c}{suffix}" for c in batch_codes]
        pct  = i / total
        text = f"掃描中... {i+1}–{min(i+batch_size, total)} / {total} 支"
        pb.progress(pct, text=text)

        try:
            multi = fetch_multi(tuple(batch_syms), period)
        except Exception:
            continue

        if suffix == ".TWO":
            time.sleep(0.5)

        for code, sym in zip(batch_codes, batch_syms):
            try:
                sdf = _extract_one(multi, sym, len(batch_syms))
                if sdf is None:
                    continue
                row = _analyze_one(code, sdf, fast_ma, slow_ma, stop_pct, bench_df=bench_df,
                                   sector=sector_map.get(code, "—") if sector_map else "—",
                                   use_atr_stop=use_atr_stop, atr_mult=atr_mult,
                                   revenue_yoy=(revenue_dict or {}).get(code))
                if row:
                    results.append(row)
            except Exception:
                continue

    pb.progress(1.0, text="掃描完成")
    return pd.DataFrame(results)


# ── 興櫃升板雷達 函式 ─────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_mops_listing_news(days_back=180):
    """從 MOPS 搜尋興櫃公司近期申請上市/上櫃、現金增資承銷相關重大訊息"""
    from datetime import datetime, timedelta
    import urllib3
    urllib3.disable_warnings()

    end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=days_back)

    sess = requests.Session()
    sess.verify = False
    base_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "zh-TW,zh;q=0.9",
    }
    # 先 GET 建立 session cookie
    try:
        sess.get("https://mops.twse.com.tw/mops/web/t05sr0100",
                 headers=base_headers, timeout=10)
    except Exception:
        pass

    url = "https://mops.twse.com.tw/mops/web/ajax_t05sr0100"
    all_rows = []

    for keyword in ["上市(櫃)前公開承銷", "申請上市", "申請上櫃"]:
        try:
            payload = {
                "encodeURIComponent": "1",
                "step": "1",
                "TYPEK": "em",
                "keyword": keyword,
                "isnew": "false",
                "SubmitButton": "查詢",
                "b_date": start_dt.strftime("%Y%m%d"),
                "e_date": end_dt.strftime("%Y%m%d"),
            }
            post_headers = {
                **base_headers,
                "Referer": "https://mops.twse.com.tw/mops/web/t05sr0100",
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                "X-Requested-With": "XMLHttpRequest",
            }
            r = sess.post(url, data=payload, headers=post_headers, timeout=25)
            r.encoding = "utf-8"
            tables = pd.read_html(r.text)
            for t in tables:
                if len(t.columns) >= 3 and len(t) > 1:
                    t.columns = [str(c) for c in t.columns]
                    t["關鍵字"] = keyword
                    all_rows.append(t)
        except Exception:
            continue

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True).drop_duplicates()


@st.cache_data(ttl=43200)
def fetch_twse_listing_applicants():
    """TWSE 官方 API：目前申請上市審查中的公司"""
    try:
        r = requests.get(
            "https://www.twse.com.tw/rwd/zh/listed/applyList?response=json",
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        data = r.json()
        if data.get("stat") == "OK" and data.get("data"):
            return pd.DataFrame(data["data"], columns=data.get("fields", []))
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=43200)
def fetch_tpex_listing_applicants():
    """TPEX 官方頁面：目前申請上櫃審查中的公司"""
    import urllib3
    urllib3.disable_warnings()
    try:
        r = requests.get(
            "https://www.tpex.org.tw/web/stock/listingApplication/lA01_01.php?l=zh-tw",
            timeout=15, verify=False,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        r.encoding = "utf-8"
        tables = pd.read_html(r.text)
        for t in tables:
            if len(t) > 1 and len(t.columns) >= 3:
                return t
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=300)
def fetch_em_price_list():
    """從 TPEX 新版 API 抓取興櫃股票當日行情表"""
    url = "https://www.tpex.org.tw/www/zh-tw/emerging/latest"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": "https://www.tpex.org.tw/zh-tw/esb/trading/info/pricing.html",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    }
    try:
        import urllib3; urllib3.disable_warnings()
        r = requests.post(url, data="id=&response=json", headers=headers, timeout=15, verify=False)
        j = r.json()
        tables = j.get("tables", [])
        if not tables:
            return pd.DataFrame()
        tbl = tables[0]
        df = pd.DataFrame(tbl["data"], columns=tbl["fields"])
        # 統一欄位名稱供下方 metric 使用
        df = df.rename(columns={"成交": "收盤", "日最高": "最高", "日最低": "最低"})
        df["漲跌"] = pd.to_numeric(df["收盤"], errors="coerce") - pd.to_numeric(df["前日均價"], errors="coerce")
        df["漲跌"] = df["漲跌"].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "—")
        df["代號"] = df["代號"].astype(str).str.strip()
        return df
    except Exception:
        return pd.DataFrame()


# ── 主畫面 ────────────────────────────────────────────────────────────────────
stock_codes = [s.strip() for s in stocks_input.split(",") if s.strip()]

if not stock_codes:
    st.warning("請在左側輸入至少一支股票代號")
    st.stop()

tabs = st.tabs([f"  {code}  " for code in stock_codes] + ["  選股掃描  ", "  興櫃升板雷達  ", "  回測  ", "  模擬交易  "])

for tab, code in zip(tabs, stock_codes):
    with tab:
        symbol = f"{code}{suffix}"

        with st.spinner(f"下載 {symbol}..."):
            try:
                raw_df = fetch(symbol, period)
            except Exception as e:
                st.error(f"下載 {symbol} 失敗：{e}")
                continue

        if raw_df is None or raw_df.empty:
            st.error(f"找不到 {symbol}，請確認代號與交易所選項是否正確")
            continue

        raw_df.index = pd.to_datetime(raw_df.index)
        df, fast_col, slow_col, supports = calc_indicators(
            raw_df, fast_ma, slow_ma, pivot_window, support_lookback
        )

        fig = build_chart(df, symbol, fast_col, slow_col, fast_ma, slow_ma, supports, stop_pct,
                          show_bb=show_bb, use_atr_stop=use_atr_stop, atr_mult=atr_mult,
                          momentum_ind=momentum_ind)

        # ── 一眼看懂 ────────────────────────────────────────────────────────────
        latest  = df.iloc[-1]
        close   = float(latest["Close"])
        gap_now = float(latest[fast_col]) - float(latest[slow_col])
        sig_q   = df[df["signal"] != 0]
        days_q  = (df.index[-1] - sig_q.iloc[-1].name).days if not sig_q.empty else 999
        last_q  = int(sig_q.iloc[-1]["signal"]) if not sig_q.empty else 0
        pats_q  = detect_all_patterns(df)

        _top_div_q  = [p for p in pats_q if "頂背離" in p]
        _consol_q   = [p for p in pats_q if any(k in p for k in ["VCP", "平台底", "杯柄", "雙底"])]
        _r1m_q = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-21]) - 1) * 100 \
                 if len(df) > 21 else 0
        _r3m_q = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-63]) - 1) * 100 \
                 if len(df) > 63 else 0
        _ext_q = (_r1m_q > 25 and not _consol_q) or (_r3m_q > 60 and not _consol_q)

        _trigger_q = _extract_trigger_price(pats_q)
        _broke_q   = (_trigger_q is not None and _trigger_q * 0.98 <= close <= _trigger_q * 1.08)
        try:
            _, _smap_q = _parse_isin_page(2 if suffix == ".TW" else 4)
            _sector_q  = _smap_q.get(code, "—")
        except Exception:
            _sector_q  = "—"

        if _top_div_q:
            action_q = "不看"
        elif _ext_q:
            action_q = "不看"
        elif _consol_q and _broke_q and gap_now > 0:
            action_q = "可買"
        elif _consol_q and gap_now > 0:
            action_q = "等待進場"
        elif gap_now > 0 and last_q == 1 and days_q <= 5:
            action_q = "等待進場"
        else:
            action_q = "不看"

        try:
            _rev_yoy_q = fetch_monthly_revenue().get(code, None)
        except Exception:
            _rev_yoy_q = None

        _CYCLICAL_Q = {"建材營造業", "水泥工業", "汽車工業"}
        if action_q == "可買" and _sector_q in _CYCLICAL_Q:
            if _rev_yoy_q is None or _rev_yoy_q <= 0:
                action_q = "等待進場"

        # 突破量能確認
        if action_q == "可買":
            _vol_avg20_q = float(df["Volume"].tail(20).mean()) if "Volume" in df.columns else 0
            _vol_today_q = float(latest["Volume"]) if pd.notna(latest.get("Volume", np.nan)) else 0
            if _vol_avg20_q > 0 and _vol_today_q < _vol_avg20_q * 1.5:
                action_q = "等待進場"

        # 收法確認
        if action_q == "可買":
            _h_q = float(latest["High"])  if pd.notna(latest.get("High",  np.nan)) else close
            _l_q = float(latest["Low"])   if pd.notna(latest.get("Low",   np.nan)) else close
            _o_q = float(latest["Open"])  if pd.notna(latest.get("Open",  np.nan)) else close
            _rng_q = _h_q - _l_q
            _pos_ok_q = (_rng_q <= 0) or ((close - _l_q) / _rng_q >= 0.6)
            if not (_pos_ok_q and close >= _o_q):
                action_q = "等待進場"

        # 細分「等待進場」語意（同 _analyze_one 邏輯）
        if action_q == "等待進場" and _consol_q:
            if _broke_q:
                action_q = "條件未足"
            elif _trigger_q is not None and close > _trigger_q * 1.08:
                action_q = "已錯過"
            else:
                action_q = "等突破"

        if use_atr_stop and "ATR14" in df.columns:
            atr_now = float(latest["ATR14"])
            stop_q  = round(close - atr_mult * atr_now, 2)
            stop_label_q = f"ATR×{atr_mult} ({atr_now:.2f})"
            stop_delta_q = f"{(stop_q/close-1)*100:+.1f}%"
        else:
            stop_q  = round(close * (1 - stop_pct / 100), 2)
            stop_label_q = f"-{stop_pct}%"
            stop_delta_q = f"-{stop_pct}%"
        resist_q = [s for s in supports if s > close]
        target_q = round(min(resist_q), 2) if resist_q else round(close * 1.15, 2)
        target_pct = round((target_q / close - 1) * 100, 1)
        rsi_q   = round(float(latest["RSI14"]), 1) if "RSI14" in df.columns and pd.notna(latest.get("RSI14")) else None
        _bench_q = _fetch_benchmark(period)
        rs_q     = _calc_rs(df, _bench_q)
        score_q  = calc_score(df, pats_q, fast_col, slow_col, rs_q, _rev_yoy_q)
        rr_q    = round((target_q - close) / (close - stop_q), 2) if 0 < stop_q < close else None

        st.markdown("---")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        if action_q == "可買":
            c1.success("**✅ 可買（訊號確認）**")
        elif action_q == "條件未足":
            c1.warning("**⚠️ 條件未足（在窗口）**")
        elif action_q == "等突破":
            c1.info("**⚡ 等突破**")
        elif action_q == "已錯過":
            c1.warning("**⏭ 已錯過**")
        elif action_q == "等待進場":
            c1.warning("**⏳ 等待進場**")
        else:
            c1.error("**❌ 不看**")
        c2.metric("進場價（參考）", f"{close:.2f}", "現價附近")
        c3.metric("止損價", f"{stop_q}", stop_delta_q, delta_color="inverse")
        rr_text = f"RR 1 : {rr_q:.1f}" if rr_q else "RR n/a"
        c4.metric("目標 / RR", f"{target_q}", f"+{target_pct}%  |  {rr_text}")
        if rsi_q is not None:
            rsi_state = "超買" if rsi_q > 70 else ("超賣" if rsi_q < 30 else "正常")
            c5.metric("RSI(14)", f"{rsi_q}", rsi_state, delta_color="off")
        c6.metric("評分", f"{score_q} / 100")

        _tdq = [p for p in pats_q if "頂背離" in p]
        _ppq = [p for p in pats_q if "頂背離" not in p]
        if _tdq:
            st.error("頂背離警告（追高風險）：" + "、".join(_tdq))
        if _ppq:
            st.info("偵測型態：" + "、".join(_ppq))
        st.markdown("---")

        # ── 財務健康 ────────────────────────────────────────────────────────────
        with st.spinner("查詢財務數據..."):
            fin = _get_fin_health(symbol)
        _sector_label = _sector_q
        grade = fin["grade"]
        _FIN_STYLE = {
            "良好":   ("success", "✅ 財務良好"),
            "普通":   ("info",    "➖ 財務普通"),
            "衰退":   ("warning", "⚠️ 盈餘衰退"),
            "虧損":   ("error",   "🚨 財務虧損"),
            "無資料": ("info",    "— 財務無資料"),
        }
        _fn_method, _fn_label = _FIN_STYLE.get(grade, ("info", grade))
        _exp_title = f"財務健康：{_fn_label}"
        if _sector_label and _sector_label not in ("—", ""):
            _exp_title += f"　｜　產業：{_sector_label}"
        with st.expander(_exp_title, expanded=(grade in ("虧損", "衰退"))):
            _fc = st.columns(5)
            getattr(_fc[0], _fn_method)(_fn_label)
            _fc[1].metric("EPS（近12月）",   f"{fin['eps']:.2f}"  if fin["eps"] is not None else "—")
            _fc[2].metric("盈餘成長（YoY）", f"{fin['eg']:+.1f}%" if fin["eg"]  is not None else "—")
            _fc[3].metric("營收成長（YoY）", f"{fin['rg']:+.1f}%" if fin["rg"]  is not None else "—")
            _fc[4].metric("ROE",             f"{fin['roe']:.1f}%" if fin["roe"] is not None else "—")
            if fin["pe"] is not None or fin["pb"] is not None:
                _fc2 = st.columns(4)
                if fin["pe"] is not None:
                    _fc2[0].metric("本益比（PE）",      f"{fin['pe']:.1f}x",
                                   help="股價 ÷ 近12月EPS，合理區間通常 10–25x")
                if fin["pb"] is not None:
                    _fc2[1].metric("股價淨值比（PB）",  f"{fin['pb']:.2f}x",
                                   help="< 1 代表股價低於帳面價值")
            if grade == "虧損":
                st.error("此股財務虧損，技術買訊可靠性下降，進場前確認是否有轉機題材。")
            elif grade == "衰退":
                st.warning("盈餘或營收明顯衰退，技術訊號需搭配轉機題材才建議操作。")

        st.plotly_chart(fig, use_container_width=True)

        # ── 動能報酬 ────────────────────────────────────────────────────────────
        mom = _calc_momentum(df)
        max_dd = _calc_max_drawdown(df)
        m_cols = st.columns(4)
        for i, (key, label) in enumerate([("R1M", "近1月"), ("R3M", "近3月"), ("R6M", "近6月")]):
            val = mom.get(key)
            if val is not None:
                clr = "normal" if val >= 0 else "inverse"
                m_cols[i].metric(label, f"{val:+.1f}%", delta_color=clr)
            else:
                m_cols[i].metric(label, "—")
        m_cols[3].metric("最大回撤（期間）", f"-{max_dd}%", delta_color="inverse")

        st.subheader("均線位置 / 乖離率（BIAS）")
        cols = st.columns(4)
        for i, ma in enumerate([5, 10, 20, 60]):
            val   = float(latest[f"MA{ma}"])
            bias  = round((close / val - 1) * 100, 2) if val else 0
            clr   = "normal" if bias >= 0 else "inverse"
            cols[i].metric(f"MA{ma}（{val:.2f}）", f"BIAS {bias:+.2f}%", delta_color=clr)

        # 相對強度 vs 0050 + 多空排列
        bench_df = _fetch_benchmark(period)
        rs_val   = _calc_rs(df, bench_df)
        gap_now  = float(latest[fast_col]) - float(latest[slow_col])

        row_info = st.columns(3)
        with row_info[0]:
            if gap_now > 0:
                st.success(f"MA{fast_ma} > MA{slow_ma}　→　多頭排列")
            else:
                st.error(f"MA{fast_ma} < MA{slow_ma}　→　空頭排列")
        with row_info[1]:
            if rs_val is not None:
                clr = "normal" if rs_val >= 0 else "inverse"
                rs_label = "跑贏大盤" if rs_val >= 0 else "跑輸大盤"
                st.metric(f"相對強度 vs 0050（近3月）", f"{rs_val:+.1f}%",
                          rs_label, delta_color=clr)
        with row_info[2]:
            _regime = _detect_regime(bench_df)  # 用 0050 大盤資料，不是個股
            _REGIME_STYLE = {
                "多頭": ("success", "多頭 — 趨勢順風"),
                "橫盤": ("info",    "橫盤 — 謹慎操作"),
                "空頭": ("error",   "空頭 ⚠️ 暫緩新進場"),
            }
            _fn, _txt = _REGIME_STYLE.get(_regime, ("info", _regime))
            getattr(st, _fn)(f"GMM 市場狀態：{_txt}")

        # ── 進場策略 + 失效條件 ──────────────────────────────────────────────────
        _, _, fails = _build_strategy(
            close, stop_q, supports, fast_col, slow_col, df, fast_ma, slow_ma
        )
        with st.expander("進場策略與失效條件", expanded=True):
            if action_q == "不看":
                st.warning("目前訊號為「不看」，不建議進場。留意以下失效條件作為參考。")
                for f in fails:
                    st.error(f)
            else:
                agg, cons, _ = _build_strategy(
                    close, stop_q, supports, fast_col, slow_col, df, fast_ma, slow_ma
                )
                sa, sb = st.columns(2)
                with sa:
                    st.markdown("**積極進場**")
                    st.info(agg)
                    st.markdown("**保守進場**")
                    st.success(cons)
                with sb:
                    st.markdown("**失效條件**")
                    for f in fails:
                        st.error(f)

        col_l, col_r = st.columns(2)

        # 支撐表
        with col_l:
            if supports:
                st.subheader("支撐區間")
                rows = []
                for lvl in sorted(supports):
                    dist  = (close - lvl) / close * 100
                    label = "下方支撐" if lvl < close else "上方壓力"
                    rows.append({"價位": f"{lvl:.2f}", "距收盤": f"{dist:.1f}%", "位置": label})
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # 近期訊號表（含止損價）
        with col_r:
            sig_rows = df[df["signal"] != 0].tail(10).copy()
            if not sig_rows.empty:
                st.subheader("近期交叉訊號")
                sig_rows["日期"]  = sig_rows.index.strftime("%Y-%m-%d")
                sig_rows["訊號"]  = sig_rows["signal"].map({1: "入場", -1: "出場"})
                sig_rows["收盤"]  = sig_rows["Close"].round(2)
                sig_rows["止損價"] = sig_rows.apply(
                    lambda r: f"{float(r['Close']) * (1 - stop_pct / 100):.2f}"
                    if r["signal"] == 1 else "—",
                    axis=1
                )
                sig_rows[f"MA{fast_ma}"] = sig_rows[fast_col].round(2)
                sig_rows[f"MA{slow_ma}"] = sig_rows[slow_col].round(2)
                st.dataframe(
                    sig_rows[["日期", "訊號", "收盤", "止損價",
                               f"MA{fast_ma}", f"MA{slow_ma}"]].reset_index(drop=True),
                    use_container_width=True, hide_index=True
                )

# ── 選股掃描頁 ────────────────────────────────────────────────────────────────
with tabs[-4]:
    st.subheader("選股掃描")

    scan_mode = st.radio(
        "掃描範圍",
        ["全市場上市（TWSE，約 900 支）", "全市場上櫃（TPEX，約 800 支）", "自訂清單"],
        horizontal=True
    )

    if scan_mode == "自訂清單":
        watchlist_input = st.text_area(
            "掃描清單（逗號或換行分隔）",
            value=DEFAULT_WATCHLIST,
            height=100
        )
        scan_source_key = watchlist_input
    else:
        watchlist_input = ""
        scan_source_key = scan_mode

    use_fin_filter = st.toggle(
        "加入財務篩選（對可買/等待進場候選股查詢 EPS / 成長率，需額外 1–3 分鐘）",
        value=False, key="use_fin_filter"
    )
    use_inst_filter = st.toggle(
        "加入法人籌碼（外資連續買超優先排序）",
        value=False, key="use_inst_filter"
    )
    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        force_rescan = st.button("重新掃描", use_container_width=True)
    with col_info:
        st.info(f"快線 MA{fast_ma} vs 慢線 MA{slow_ma}，止損 {stop_pct}%，期間 {period}")

    scan_key = f"{scan_source_key}|{suffix}|{period}|{fast_ma}|{slow_ma}|{stop_pct}|atr={use_atr_stop}|fin={use_fin_filter}|inst={use_inst_filter}"
    need_scan = (
        force_rescan
        or "scan_result" not in st.session_state
        or st.session_state.get("scan_key") != scan_key
    )

    if need_scan:
        if scan_mode == "全市場上市（TWSE，約 900 支）":
            with st.spinner("從證交所抓股票清單..."):
                scan_codes = fetch_twse_codes()
            if not scan_codes:
                st.error("無法取得 TWSE 股票清單，請確認網路連線")
                st.stop()
        elif scan_mode == "全市場上櫃（TPEX，約 800 支）":
            with st.spinner("從櫃買中心抓股票清單..."):
                scan_codes = fetch_tpex_codes()
            if not scan_codes:
                st.error("無法取得 TPEX 股票清單，請確認網路連線")
                st.stop()
        else:
            scan_codes = [
                s.strip()
                for s in watchlist_input.replace("\n", ",").split(",")
                if s.strip()
            ]

        if not scan_codes:
            st.warning("掃描清單是空的")
        else:
            inst_data = {}
            if use_inst_filter:
                with st.spinner("取得三大法人籌碼資料（TWSE T86）..."):
                    inst_data = _fetch_institutional_data()
            st.caption(f"共 {len(scan_codes)} 支，每 100 支一批下載，請稍候...")
            pb = st.progress(0, text="準備中...")
            _bench       = _fetch_benchmark(period)
            _sector_map  = fetch_sector_map(suffix)
            with st.spinner("取得月營收資料..."):
                _rev_dict = fetch_monthly_revenue()
            result_df = scan_stocks(scan_codes, suffix, period, fast_ma, slow_ma, stop_pct, pb,
                                    bench_df=_bench, sector_map=_sector_map,
                                    use_atr_stop=use_atr_stop, atr_mult=atr_mult,
                                    revenue_dict=_rev_dict)
            pb.empty()
            # ── GMM 大盤狀態 banner ──────────────────────────────────────────
            if _bench is not None and not _bench.empty:
                _scan_regime = _detect_regime(_bench)
                if _scan_regime == "空頭":
                    st.error("大盤 GMM：空頭 — 可買訊號可信度降低，建議觀望或縮減部位")
                    if "操作" in result_df.columns:
                        result_df.loc[result_df["操作"] == "可買", "操作"] = "條件未足"
                elif _scan_regime == "橫盤":
                    st.warning("大盤 GMM：橫盤 — 個股勝率下降，謹慎操作")
                else:
                    st.success("大盤 GMM：多頭 — 趨勢順風")
            # RS vs 產業（方案二：同類股掃描結果中位數）
            if "產業" in result_df.columns and "3M%" in result_df.columns:
                def _pct_float(s):
                    try:
                        return float(str(s).replace("%", "").replace("+", ""))
                    except Exception:
                        return np.nan
                _tmp = result_df["3M%"].apply(_pct_float)
                _sec_med = result_df.assign(_3m=_tmp).groupby("產業")["_3m"].median()
                def _fmt_rs_sector(v, s):
                    if pd.isna(v) or s not in _sec_med:
                        return "—"
                    med = _sec_med.get(s)
                    if pd.isna(med):
                        return "—"
                    diff = v - med
                    return f"{diff:+.1f}%" if not np.isnan(diff) else "—"
                result_df["RS vs產業"] = [
                    _fmt_rs_sector(v, s)
                    for v, s in zip(_tmp, result_df["產業"])
                ]
            # ── 法人籌碼合併 ────────────────────────────────────────────────────
            if use_inst_filter and inst_data:
                    result_df["連買天"] = result_df["代號"].map(
                        lambda c: inst_data.get(c, {}).get("consec", 0)
                    )
                    result_df["外資累計(張)"] = result_df["代號"].map(
                        lambda c: inst_data.get(c, {}).get("total", 0) // 1000
                    )

            if scan_mode != "自訂清單" and "操作" in result_df.columns:
                from datetime import datetime as _dtnow
                _save_scan_history(
                    _dtnow.now().strftime("%Y-%m-%d"),
                    result_df[result_df["操作"].isin(["可買", "條件未足", "等突破", "等待進場"])]["代號"].tolist()
                )
            st.session_state["scan_result"] = result_df
            st.session_state["scan_key"] = scan_key
            if use_fin_filter and not result_df.empty and "操作" in result_df.columns:
                cands = result_df[result_df["操作"].isin(["可買", "條件未足", "等突破", "等待進場"])]
                if not cands.empty:
                    fin_pb = st.progress(0, text="查詢財務數據...")
                    grade_map = {}
                    rg_map = {}
                    for _fi, (_idx, _row) in enumerate(cands.iterrows()):
                        fh = _get_fin_health(f"{_row['代號']}{suffix}")
                        grade_map[_idx] = fh["grade"]
                        rg_map[_idx] = fh["rg"]  # 年度營收成長 %
                        fin_pb.progress((_fi + 1) / len(cands),
                                        text=f"財務 {_fi+1}/{len(cands)}: {_row['代號']}")
                    fin_pb.empty()
                    _GRADE_EMOJI = {"良好": "✅ 良好", "普通": "➖ 普通",
                                    "衰退": "⚠️ 衰退", "虧損": "🚨 虧損"}
                    result_df["財務"] = result_df.index.map(
                        lambda x: _GRADE_EMOJI.get(grade_map.get(x, ""), "—")
                    )
                    # 催化劑：年度營收成長 >= 30% 且分數被 RSI 上限壓到 <= 40 → 放寬至最多 70
                    def _adjust_catalyst(row):
                        rg = rg_map.get(row.name)
                        rsi = row.get("RSI")
                        try:
                            rsi_val = float(rsi) if rsi not in (None, "—") else 0
                        except (ValueError, TypeError):
                            rsi_val = 0
                        if rg is not None and rg >= 30 and rsi_val >= 78 and row["分數"] <= 40:
                            return min(70, int(row["分數"] * 1.75))
                        return row["分數"]
                    result_df["分數"] = result_df.apply(_adjust_catalyst, axis=1)
                    result_df["催化劑"] = result_df.index.map(
                        lambda x: "✅ 業績強" if (rg_map.get(x) is not None and rg_map.get(x) >= 30) else "—"
                    )
                    st.session_state["scan_result"] = result_df
    else:
        result_df = st.session_state.get("scan_result", pd.DataFrame())

    if not result_df.empty:
        if "財務" in result_df.columns:
            if st.checkbox("排除財務虧損股", value=True, key="excl_fin_risk"):
                result_df = result_df[~result_df["財務"].str.contains("虧損", na=False)]
        if "連掃天" not in result_df.columns:
            from datetime import datetime as _dtnow
            _today_str = _dtnow.now().strftime("%Y-%m-%d")
            result_df["連掃天"] = result_df["代號"].apply(
                lambda c: _count_streak(c, _load_scan_history(), _today_str)
            )

        order = {"近期黃金交叉": 0, "多頭持續中": 1, "觀察中": 2, "近期死亡交叉": 3, "空頭排列": 4}
        result_df["_sort"] = result_df["狀態"].map(order).fillna(9)
        result_df["_days"] = pd.to_numeric(result_df["幾天前"], errors="coerce").fillna(9999)
        # 同組內按分數降序排
        result_df = result_df.sort_values(["_sort", "分數", "_days"],
                                          ascending=[True, False, True]).drop(columns=["_sort", "_days"])

        entry_df = result_df[result_df["狀態"] == "近期黃金交叉"]
        bull_df  = result_df[result_df["狀態"] == "多頭持續中"]
        watch_df = result_df[result_df["狀態"] == "觀察中"]
        exit_df  = result_df[result_df["狀態"].isin(["近期死亡交叉", "空頭排列"])]

        # ── 簡易操作清單（最頂端）────────────────────────────────────────────────
        _ALL_ACTIONS = ["可買", "條件未足", "等突破", "等待進場"]
        _ACTION_RANK = {"可買": 0, "條件未足": 1, "等突破": 2, "等待進場": 3}
        _OP_LABEL    = {
            "可買":   "✅ 可買",
            "條件未足": "⚠️ 條件未足",
            "等突破": "⚡ 等突破",
            "等待進場": "⏳ 等待進場",
            "已錯過": "⏭ 已錯過",
        }

        def _trigger_dist(row):
            try:
                tv = float(str(row.get("觸發價", "0")))
                c  = float(row.get("收盤價", 0))
                return (tv - c) / tv if tv > 0 else 1.0
            except Exception:
                return 1.0

        if "操作" in result_df.columns:
            easy_df = result_df[result_df["操作"].isin(_ALL_ACTIONS)].copy()
            if not easy_df.empty:
                # 排序：可買/條件未足 依連買天↓分數↓；等突破 依距觸發距離↑；等待進場 依分數↓
                easy_df["_rank"]  = easy_df["操作"].map(_ACTION_RANK).fillna(9)
                easy_df["_dist"]  = easy_df.apply(_trigger_dist, axis=1)
                _consec = pd.to_numeric(easy_df["連買天"], errors="coerce").fillna(0) \
                          if "連買天" in easy_df.columns else pd.Series(0, index=easy_df.index)
                easy_df["_consec"] = _consec
                easy_df["_key2"] = easy_df.apply(
                    lambda r: r["_dist"] if r["操作"] == "等突破" else -r["_consec"],
                    axis=1
                )
                easy_df = easy_df.sort_values(
                    ["_rank", "_key2", "分數"], ascending=[True, True, False]
                ).drop(columns=["_rank", "_dist", "_consec", "_key2"])

                _top_n = st.slider("顯示前幾名", 10, 50, 20, key="top_n_slider")
                st.markdown(f"### 操作建議清單（前 {_top_n} 名）")
                st.caption("✅可買 → ⚠️條件未足（在窗口）→ ⚡等突破（最近觸發優先）→ ⏳等待進場（純均線）｜⏭已錯過 不顯示")
                easy_df = easy_df.head(_top_n)
                show_cols = [c for c in ["分數", "操作", "財務", "催化劑", "連掃天", "連買天", "外資累計(張)",
                                          "代號", "產業", "收盤價", "止損價", "觸發價",
                                          "RR", "RSI", "RS vs大盤", "RS vs產業", "1M%", "3M%",
                                          "距52週高%", "偵測型態", "訊號日期"]
                             if c in easy_df.columns]
                disp = easy_df[show_cols].reset_index(drop=True)
                disp["操作"] = disp["操作"].map(_OP_LABEL).fillna(disp["操作"])
                st.dataframe(disp, use_container_width=True, hide_index=True)
                st.download_button(
                    "下載操作清單 CSV",
                    data=disp.to_csv(index=False, encoding="utf-8-sig"),
                    file_name="scan_candidates.csv",
                    mime="text/csv",
                    key="dl_candidates",
                )
                st.divider()

        # ── 型態候選股（優先顯示）────────────────────────────────────────────────
        try:
            pattern_df = result_df[result_df["偵測型態"] != "—"].copy()
        except KeyError:
            pattern_df = pd.DataFrame()
        if not pattern_df.empty:
            st.markdown("#### 型態偵測候選股（VCP / 雙底 / 平台底 / 杯柄）")
            PATTERN_ORDER = {"VCP": 0, "杯柄": 1, "雙底": 2, "平台底": 3}
            def _pat_rank(s):
                for k, v in PATTERN_ORDER.items():
                    if k in str(s):
                        return v
                return 9
            pattern_df["_pr"] = pattern_df["偵測型態"].apply(_pat_rank)
            pattern_df = pattern_df.sort_values("_pr").drop(columns="_pr")
            st.dataframe(
                pattern_df[["代號", "收盤價", "偵測型態", "狀態", "排列", "訊號日期"]].reset_index(drop=True),
                use_container_width=True, hide_index=True,
            )
            st.divider()

        if not entry_df.empty:
            st.markdown("#### 近期黃金交叉（最值得關注）")
            st.dataframe(entry_df.drop(columns="排列", errors="ignore").reset_index(drop=True),
                         use_container_width=True, hide_index=True)

        if not bull_df.empty:
            st.markdown("#### 多頭排列持續中")
            st.dataframe(bull_df.drop(columns="排列", errors="ignore").reset_index(drop=True),
                         use_container_width=True, hide_index=True)

        if not watch_df.empty:
            with st.expander(f"觀察中（{len(watch_df)} 支）"):
                st.dataframe(watch_df.drop(columns="排列", errors="ignore").reset_index(drop=True),
                             use_container_width=True, hide_index=True)

        if not exit_df.empty:
            with st.expander(f"空頭 / 近期死亡交叉（{len(exit_df)} 支，避開）"):
                st.dataframe(exit_df.drop(columns="排列", errors="ignore").reset_index(drop=True),
                             use_container_width=True, hide_index=True)

        st.caption(f"共掃描 {len(result_df)} 支，資料來自 Yahoo Finance，僅供參考，不構成投資建議。")
        st.download_button(
            "下載完整掃描結果 CSV",
            data=result_df.to_csv(index=False, encoding="utf-8-sig"),
            file_name="scan_full.csv",
            mime="text/csv",
            key="dl_full",
        )

# ── 興櫃升板雷達頁 ────────────────────────────────────────────────────────────
with tabs[-3]:
    st.subheader("興櫃升板雷達")
    st.caption("複製 6959 模式：從 MOPS 找正在申請升板的興櫃股")

    with st.expander("為什麼這樣找？（6959 覆盤）"):
        st.markdown("""
**6959 兆捷科技 實際走勢：**

| 日期 | 事件 | 意義 |
|------|------|------|
| 2026/3/11 | 公告「辦理現金增資作為上市(櫃)前公開承銷」| **核心訊號** |
| 2026/4/10 | 受邀參加 QIC CEO Week（法人投資週）| 機構開始注意 |
| 2026/3/4 | 年度低點 65.8 | 好的卡位時機 |
| 2026/4/20 | 最高點 220.5 | 漲幅 3.4 倍，6 週 |

**篩選條件：**
1. 興櫃市場，公告「現金增資作為上市/上櫃前公開承銷」
2. 股價在近期低點（距52週高點跌幅 > 30%）
3. 有產業題材（AI / 半導體 / 特殊材料 / 國防等）
4. 開始被邀請參加法人說明會
        """)

    st.divider()

    # ── MOPS 公告掃描 ──────────────────────────────────────────────────────────
    st.markdown("#### MOPS 重大訊息掃描")
    col_days, col_btn = st.columns([3, 1])
    with col_days:
        em_days = st.slider("搜尋近幾天的公告", 30, 365, 180, key="em_days")
    with col_btn:
        st.write("")
        do_em_scan = st.button("掃描", use_container_width=True, key="em_scan_btn")

    if do_em_scan or "em_scan_done" in st.session_state:
        if do_em_scan:
            with st.spinner("查詢 MOPS 重大訊息中..."):
                em_news = fetch_mops_listing_news(em_days)
            with st.spinner("查詢 TWSE 申請上市名單..."):
                twse_apply = fetch_twse_listing_applicants()
            with st.spinner("查詢 TPEX 申請上櫃名單..."):
                tpex_apply = fetch_tpex_listing_applicants()
            st.session_state["em_news"]      = em_news
            st.session_state["twse_apply"]   = twse_apply
            st.session_state["tpex_apply"]   = tpex_apply
            st.session_state["em_scan_done"] = True
        else:
            em_news    = st.session_state.get("em_news",    pd.DataFrame())
            twse_apply = st.session_state.get("twse_apply", pd.DataFrame())
            tpex_apply = st.session_state.get("tpex_apply", pd.DataFrame())

        any_data = False

        # ── MOPS 公告 ──
        if not em_news.empty:
            any_data = True
            st.markdown("##### MOPS 重大訊息（興櫃申請升板相關）")
            st.dataframe(em_news, use_container_width=True, hide_index=True)
        else:
            st.warning("MOPS 重大訊息：無資料（網站可能需要登入或 IP 限制）")

        # ── TWSE 申請上市 ──
        if not twse_apply.empty:
            any_data = True
            st.markdown("##### TWSE 申請上市審查中（這些公司正在走上市程序）")
            st.dataframe(twse_apply, use_container_width=True, hide_index=True)
        else:
            st.warning("TWSE 申請上市名單：無資料")

        # ── TPEX 申請上櫃 ──
        if not tpex_apply.empty:
            any_data = True
            st.markdown("##### TPEX 申請上櫃審查中（這些公司正在走上櫃程序）")
            st.dataframe(tpex_apply, use_container_width=True, hide_index=True)
        else:
            st.warning("TPEX 申請上櫃名單：無資料")

        if not any_data:
            st.error("三個來源都無資料，可能是網路問題。請直接用下方連結手動查：")
        st.info(
            "手動查詢連結：\n"
            "- [MOPS 重大訊息](https://mops.twse.com.tw/mops/web/t05sr0100)"
            "（市場別選「興櫃」，關鍵字輸入「上市(櫃)前公開承銷」）\n"
            "- [TWSE 申請上市名單](https://www.twse.com.tw/zh/listed/applyList.html)\n"
            "- [TPEX 申請上櫃名單](https://www.tpex.org.tw/web/stock/listingApplication/lA01_01.php?l=zh-tw)"
        )

    st.divider()

    # ── 興櫃即時報價 ───────────────────────────────────────────────────────────
    st.markdown("#### 興櫃股即時報價（TPEX）")
    col_input, col_query = st.columns([3, 1])
    with col_input:
        em_code_input = st.text_input("輸入興櫃股代號", placeholder="例：6959", key="em_price_code")
    with col_query:
        st.write("")
        do_em_price = st.button("查詢", use_container_width=True, key="em_price_btn")

    if do_em_price and em_code_input:
        with st.spinner(f"從 TPEX 查詢 {em_code_input} 報價..."):
            em_price_df = fetch_em_price_list()

        if em_price_df.empty:
            st.warning("TPEX API 無回應，請直接前往 TPEX 網站查詢")
        else:
            match = em_price_df[em_price_df["代號"].astype(str) == str(em_code_input)]
            if match.empty:
                st.warning(f"找不到 {em_code_input}，請確認是否為今日有成交的興櫃股")
                with st.expander("今日所有興櫃報價"):
                    st.dataframe(em_price_df, use_container_width=True, hide_index=True)
            else:
                row = match.iloc[0]
                cols4 = st.columns(4)
                cols4[0].metric("代號 / 名稱", f"{row.get('代號','')} {row.get('名稱','')}")
                cols4[1].metric("收盤價", row.get("收盤", "—"))
                cols4[2].metric("漲跌", row.get("漲跌", "—"))
                cols4[3].metric("最高 / 最低", f"{row.get('最高','—')} / {row.get('最低','—')}")
                st.dataframe(match, use_container_width=True, hide_index=True)

    st.caption("資料來源：MOPS 公開資訊觀測站、TPEX 興櫃報價。僅供參考，不構成投資建議。")

# ── 回測頁 ───────────────────────────────────────────────────────────────────
with tabs[-2]:
    st.subheader("策略回測")

    bt_c1, bt_c2, bt_c3, bt_c4 = st.columns([2, 2, 2, 1])
    with bt_c1:
        bt_code = st.selectbox("股票", stock_codes, key="bt_code")
        bt_exch = suffix
    with bt_c2:
        bt_period = st.selectbox("回測期間", ["1y", "2y", "3y", "5y"], index=1, key="bt_period")
    with bt_c3:
        bt_capital = st.number_input("初始資金（元）", min_value=10_000,
                                     value=1_000_000, step=50_000, key="bt_capital")
    with bt_c4:
        st.write("")
        bt_run = st.button("執行回測", use_container_width=True, key="bt_run")

    bt_use_pattern = st.toggle(
        "型態突破策略（走步前進）",
        value=False, key="bt_use_pattern",
        help="關閉：MA 黃金交叉/死叉。開啟：VCP/杯柄/雙底/平台底 + 觸發窗口，走步前進計算，速度較慢。"
    )
    if bt_use_pattern:
        st.caption(f"型態突破策略（走步前進）｜出場：死叉 or 止損 {stop_pct}%｜含台股手續費")
    else:
        st.caption(f"均線交叉訊號（MA{fast_ma} vs MA{slow_ma}）｜含台股手續費（買 0.0855%，賣 0.3855%）")

    bt_key = (f"{bt_code}|{bt_exch}|{bt_period}|{bt_capital}"
              f"|{fast_ma}|{slow_ma}|{stop_pct}|atr={use_atr_stop}|pat={bt_use_pattern}")
    if bt_run or ("bt_result" in st.session_state and st.session_state.get("bt_key") == bt_key):
        if bt_run or "bt_result" not in st.session_state:
            bt_sym = f"{bt_code}{bt_exch}"
            spinner_msg = (f"下載 {bt_sym} 並執行型態突破回測（走步前進，需 10~30 秒）..."
                           if bt_use_pattern else f"下載 {bt_sym} 並執行回測...")
            with st.spinner(spinner_msg):
                try:
                    bt_raw = fetch(bt_sym, bt_period)
                except Exception as e:
                    st.error(f"下載失敗：{e}")
                    bt_raw = None
            if bt_raw is not None and not bt_raw.empty:
                bt_raw.index = pd.to_datetime(bt_raw.index)
                bt_df, bt_fc, bt_sc, _ = calc_indicators(
                    bt_raw, fast_ma, slow_ma, pivot_window, support_lookback
                )
                if bt_use_pattern:
                    bt_stats, bt_trades, bt_eq, bt_bh = run_pattern_backtest(
                        bt_df, bt_fc, bt_sc, fast_ma, slow_ma,
                        initial_capital=float(bt_capital),
                        use_atr_stop=use_atr_stop, atr_mult=atr_mult, stop_pct=stop_pct
                    )
                else:
                    bt_stats, bt_trades, bt_eq, bt_bh = run_backtest(
                        bt_df, bt_fc, bt_sc, fast_ma, slow_ma,
                        initial_capital=float(bt_capital),
                        use_atr_stop=use_atr_stop, atr_mult=atr_mult, stop_pct=stop_pct
                    )
                st.session_state["bt_result"] = (bt_stats, bt_trades, bt_eq, bt_bh)
                st.session_state["bt_key"] = bt_key
                st.session_state["bt_sym_used"] = bt_code
                st.session_state["bt_strategy_used"] = "pattern" if bt_use_pattern else "ma"
            else:
                st.error(f"找不到 {bt_sym}")

        if "bt_result" in st.session_state:
            bt_stats, bt_trades, bt_eq, bt_bh = st.session_state["bt_result"]
            bt_sym_used     = st.session_state.get("bt_sym_used", bt_code)
            bt_strategy_used = st.session_state.get("bt_strategy_used", "ma")
            bt_strat_label  = "型態突破" if bt_strategy_used == "pattern" else f"MA{fast_ma}×MA{slow_ma}"

            # ── 績效指標 ──────────────────────────────────────────────────────
            r1, r2, r3, r4, r5, r6 = st.columns(6)
            _tr  = bt_stats["總報酬%"]
            _bhr = bt_stats["買持報酬%"]
            _exc = bt_stats["超額報酬%"]
            r1.metric("總報酬", f"{_tr:+.1f}%",
                      f"買持 {_bhr:+.1f}%",
                      delta_color="normal" if _tr >= 0 else "inverse")
            r2.metric("超額報酬", f"{_exc:+.1f}%",
                      delta_color="normal" if _exc >= 0 else "inverse")
            r3.metric("年化 Sharpe", f"{bt_stats['年化Sharpe']:.2f}")
            r4.metric("最大回撤", f"{bt_stats['最大回撤%']:.1f}%",
                      delta_color="inverse")
            r5.metric("勝率", f"{bt_stats['勝率%']:.1f}%",
                      f"{bt_stats['交易次數']} 筆")
            _pf = bt_stats["Profit Factor"]
            r6.metric("Profit Factor", f"{_pf:.2f}" if isinstance(_pf, float) else str(_pf))

            r7, r8, r9, r10 = st.columns(4)
            r7.metric("期末資金", f"{bt_stats['期末資金']:,.0f} 元")
            r8.metric("平均獲利", f"{bt_stats['平均獲利%']:+.2f}%",
                      delta_color="normal")
            r9.metric("平均虧損", f"{bt_stats['平均虧損%']:.2f}%",
                      delta_color="inverse")
            _stop_cnt = int((bt_trades["出場類型"] == "止損").sum()) if not bt_trades.empty else 0
            r10.metric("止損觸發", f"{_stop_cnt} 次")

            # ── 資金曲線 ──────────────────────────────────────────────────────
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=bt_eq.index, y=bt_eq.values,
                name=f"策略（{bt_strat_label}）",
                line=dict(color="#42A5F5", width=2)
            ))
            fig_eq.add_trace(go.Scatter(
                x=bt_bh.index, y=bt_bh.values,
                name="買進持有",
                line=dict(color="rgba(180,180,180,0.55)", width=1.5, dash="dot")
            ))
            fig_eq.update_layout(
                height=380, template="plotly_dark",
                paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                title=f"{bt_sym_used} 資金曲線（初始 {bt_capital:,.0f} 元）",
                legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_eq, use_container_width=True)

            # ── 回撤曲線 ──────────────────────────────────────────────────────
            _peak   = bt_eq.cummax()
            _dd_pct = (bt_eq - _peak) / _peak * 100
            fig_dd = go.Figure(go.Scatter(
                x=_dd_pct.index, y=_dd_pct.values,
                fill="tozeroy", fillcolor="rgba(239,83,80,0.18)",
                line=dict(color="rgba(239,83,80,0.75)", width=1),
                name="回撤%"
            ))
            fig_dd.update_layout(
                height=180, template="plotly_dark",
                paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                title="策略回撤曲線",
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_dd, use_container_width=True)

            # ── 交易明細 ──────────────────────────────────────────────────────
            if not bt_trades.empty:
                st.subheader(f"交易明細（{len(bt_trades)} 筆）")
                st.dataframe(bt_trades, use_container_width=True, hide_index=True)
                st.download_button(
                    "下載交易明細 CSV",
                    data=bt_trades.to_csv(index=False, encoding="utf-8-sig"),
                    file_name=f"backtest_{bt_sym_used}.csv",
                    mime="text/csv",
                    key="dl_bt"
                )
            else:
                st.info("回測期間內無均線交叉訊號，無法產生交易記錄")

    else:
        st.info("選好股票與參數後，點「執行回測」查看歷史策略績效")


# ── 模擬交易頁 ────────────────────────────────────────────────────────────────
with tabs[-1]:
    st.subheader("模擬交易")
    st.caption("記錄模擬持倉，追蹤即時損益。資料存於 D:\\私人\\股票分析\\paper_trades.json")

    data = _load_trades()
    positions = data.get("positions", [])
    closed = data.get("closed", [])

    # ── 新增持倉 ──────────────────────────────────────────────────────────────
    with st.expander("＋ 新增持倉", expanded=len(positions) == 0):
        def _sync_price():
            code = st.session_state.get("paper_code", "").strip()
            exch = st.session_state.get("paper_exch", ".TW")
            if code:
                p = _fetch_latest_price(f"{code}{exch}")
                if p:
                    st.session_state["paper_price"] = round(p, 2)

        cx1, cx2 = st.columns([2, 2])
        inp_code = cx1.text_input("股票代號", placeholder="2330", key="paper_code", on_change=_sync_price)
        inp_exch = cx2.radio("交易所", [".TW", ".TWO"], horizontal=True, key="paper_exch", on_change=_sync_price)

        with st.form("add_pos"):
            c3, c4 = st.columns(2)
            inp_price  = c3.number_input("買進均價", min_value=0.01, step=0.5, format="%.2f", key="paper_price")
            inp_shares = c4.number_input("張數", min_value=1, step=1, value=1)
            inp_date   = st.date_input("買進日期", value=pd.Timestamp.today())
            inp_note   = st.text_input("備註（選填）", placeholder="為什麼買？止損條件？")
            add_btn    = st.form_submit_button("新增持倉", use_container_width=True)

        if add_btn:
            if not inp_code.strip():
                st.error("請輸入股票代號")
            elif inp_price <= 0:
                st.error("請輸入買進價格")
            else:
                positions.append({
                    "id": str(_uuid.uuid4())[:8],
                    "code": inp_code.strip(),
                    "exchange": inp_exch,
                    "buy_date": str(inp_date),
                    "buy_price": float(inp_price),
                    "shares": int(inp_shares),
                    "note": inp_note.strip(),
                })
                data["positions"] = positions
                _save_trades(data)
                st.success(f"已新增 {inp_code.strip()} × {inp_shares} 張，買進價 {inp_price:.2f}")
                st.rerun()

    if not positions:
        st.info("目前沒有持倉，點上方「新增持倉」開始記錄")
    else:
        # ── 抓現價，算損益 ────────────────────────────────────────────────────
        rows = []
        total_cost = 0.0
        total_value = 0.0
        price_ok = 0

        for p in positions:
            sym = f"{p['code']}{p['exchange']}"
            cur = _fetch_latest_price(sym)
            cost_amt = p["buy_price"] * p["shares"] * 1000
            total_cost += cost_amt
            if cur is not None:
                val     = cur * p["shares"] * 1000
                pnl     = val - cost_amt
                pnl_pct = (cur / p["buy_price"] - 1) * 100
                total_value += val
                price_ok += 1
            else:
                val = pnl = pnl_pct = None

            rows.append({
                "_id":      p["id"],
                "代號":     p["code"],
                "買進日期": p["buy_date"],
                "買進價":   p["buy_price"],
                "現價":     round(cur, 2) if cur is not None else "—",
                "漲跌":     f"{cur - p['buy_price']:+.2f}" if cur is not None else "—",
                "損益%":    f"{pnl_pct:+.1f}%" if pnl_pct is not None else "—",
                "張數":     p["shares"],
                "成本(元)": f"{cost_amt:,.0f}",
                "市值(元)": f"{val:,.0f}" if val is not None else "—",
                "_pnl":     pnl,
                "備註":     p.get("note", ""),
            })

        # ── 總覽 ──────────────────────────────────────────────────────────────
        all_priced = price_ok == len(positions)
        total_pnl = (total_value - total_cost) if all_priced else None
        total_pnl_pct = (total_pnl / total_cost * 100) if (total_pnl is not None and total_cost > 0) else None

        realized = sum(
            (p["sell_price"] - p["buy_price"]) * p["shares"] * 1000 for p in closed
        ) if closed else 0.0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("持倉數", f"{len(positions)} 檔")
        m2.metric("總成本", f"{total_cost:,.0f} 元")
        if total_pnl is not None:
            m3.metric("目前市值", f"{total_value:,.0f} 元")
            m4.metric("未實現損益", f"{total_pnl:+,.0f} 元", f"{total_pnl_pct:+.1f}%")
        else:
            m3.metric("目前市值", "部分無法取得")
            m4.metric("未實現損益", "—")

        if closed:
            st.metric("已實現損益（歷史）", f"{realized:+,.0f} 元")

        st.button("重新整理報價", on_click=_fetch_latest_price.clear)

        # ── 持倉明細 ──────────────────────────────────────────────────────────
        disp_df = pd.DataFrame(rows).drop(columns=["_id", "_pnl"])
        st.dataframe(disp_df, use_container_width=True, hide_index=True)

        # ── 出場 ──────────────────────────────────────────────────────────────
        st.divider()
        st.markdown("#### 出場紀錄")
        pos_opts = {
            f"{p['code']}  買@{p['buy_price']}  ({p['buy_date']})  ×{p['shares']}張": p["id"]
            for p in positions
        }
        _default_sell = float(positions[0]["buy_price"]) if positions else 1.0
        with st.form("close_pos"):
            sel_label  = st.selectbox("選擇持倉", list(pos_opts.keys()))
            cc1, cc2   = st.columns(2)
            sell_price = cc1.number_input("賣出價格", min_value=0.01, value=_default_sell, step=0.5, format="%.2f")
            sell_date  = cc2.date_input("賣出日期", value=pd.Timestamp.today())
            close_btn  = st.form_submit_button("確認出場", use_container_width=True)

        if close_btn and sell_price > 0:
            sel_id = pos_opts[sel_label]
            pos    = next((p for p in positions if p["id"] == sel_id), None)
            if pos is None:
                st.error("找不到該部位，請重新整理頁面")
                st.stop()
            pnl    = (sell_price - pos["buy_price"]) * pos["shares"] * 1000
            closed.append({**pos, "sell_date": str(sell_date), "sell_price": float(sell_price)})
            data["positions"] = [p for p in positions if p["id"] != sel_id]
            data["closed"]    = closed
            _save_trades(data)
            _fetch_latest_price.clear()
            tag = "獲利" if pnl >= 0 else "虧損"
            st.success(f"{pos['code']} 出場（{tag}），損益 {pnl:+,.0f} 元")
            st.rerun()

    # ── 已出場紀錄 ────────────────────────────────────────────────────────────
    if closed:
        realized = sum(
            (p["sell_price"] - p["buy_price"]) * p["shares"] * 1000 for p in closed
        )
        with st.expander(f"已出場紀錄（{len(closed)} 筆，合計 {realized:+,.0f} 元）"):
            c_rows = []
            for p in closed:
                pnl     = (p["sell_price"] - p["buy_price"]) * p["shares"] * 1000
                pnl_pct = (p["sell_price"] / p["buy_price"] - 1) * 100
                c_rows.append({
                    "代號":     p["code"],
                    "買進日期": p["buy_date"],
                    "買進價":   p["buy_price"],
                    "賣出日期": p["sell_date"],
                    "賣出價":   p["sell_price"],
                    "張數":     p["shares"],
                    "損益(元)": f"{pnl:+,.0f}",
                    "損益%":    f"{pnl_pct:+.1f}%",
                    "備註":     p.get("note", ""),
                })
            st.dataframe(pd.DataFrame(c_rows), use_container_width=True, hide_index=True)
