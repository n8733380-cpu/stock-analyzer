"""
daily_scan.py — 每日收盤後自動掃描台股，結果寄至 Gmail
建議排程時間：18:00（T86 法人資料約 17:30 發布）
執行時間：約 10-15 分鐘（TWSE 全市場約 900 支）
"""
import os, sys, smtplib, time, requests, re, json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import concurrent.futures
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import yfinance as yf
import pandas as pd
import numpy as np

# ── 設定 ─────────────────────────────────────────────────────────────────────
GMAIL_USER   = "n8733380@gmail.com"
GMAIL_APP_PW = os.environ.get("GMAIL_APP_PW", "")  # 從環境變數讀取
TO_EMAIL     = "n8733380@gmail.com"

FAST_MA   = 5
SLOW_MA   = 20
STOP_PCT  = 8
PERIOD    = "1y"
MIN_SCORE     = 55   # 上市門檻
MIN_SCORE_OTC = 45   # 上櫃門檻（無 T86 法人資料，分數天花板較低）
TOP_N         = 10   # 各市場各取前 N
SCAN_OTC      = True # 同時掃上櫃（.TWO）

# MOPS 重大公告監控關鍵字
MOPS_KEYWORDS = [
    "資產處分", "出售土地", "出售廠房", "出售不動產",
    "重大合約", "簽訂合作", "策略合作", "取得重大資產",
]

SCAN_HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scan_history.json")

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

# ── 進度列替代（print）────────────────────────────────────────────────────────
class _Progress:
    def progress(self, pct, text=""):
        print(f"\r  {text}", end="", flush=True)

# ── 法人籌碼（T86）────────────────────────────────────────────────────────────
def _fetch_institutional_data():
    daily = {}
    d = datetime.today()
    while len(daily) < 5 and (datetime.today() - d).days < 60:
        d -= timedelta(days=1)
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
        time.sleep(0.3)
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

# ── 股票代號清單 ───────────────────────────────────────────────────────────────
def _parse_isin_page(mode):
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

def fetch_twse_codes():
    try:
        import twstock
        codes = [
            code for code, info in twstock.codes.items()
            if getattr(info, "market", "") == "上市"
            and ((len(code) == 4 and code[0] != "0") or code.startswith("00"))
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

def fetch_otc_codes():
    """抓取上櫃股票代號"""
    try:
        import twstock
        codes = [
            code for code, info in twstock.codes.items()
            if getattr(info, "market", "") == "上櫃"
            and len(code) == 4 and code[0] != "0"
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

def fetch_monthly_revenue():
    """抓取最新月營收 YoY 成長率，回傳 {code: yoy_pct}
    上市：TWSE OpenAPI（穩定）
    上櫃：MOPS AJAX（本機可能被擋，GitHub Actions 通常可通）
    """
    result = {}

    # ── 上市：TWSE OpenAPI ────────────────────────────────────────────
    try:
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
    except Exception as e:
        print(f"  月營收上市抓取失敗：{e}")

    # ── 上櫃：MOPS AJAX（需最新一個月參數）──────────────────────────
    try:
        today = datetime.today()
        ref   = (today.replace(day=1) - timedelta(days=1)) if today.day >= 12 \
                else (today.replace(day=1) - timedelta(days=32))
        roc_year, month = ref.year - 1911, ref.month
        url  = "https://mops.twse.com.tw/mops/web/ajax_t05st10_ifrs"
        hdrs = {
            "User-Agent": "Mozilla/5.0",
            "Referer":    "https://mops.twse.com.tw/mops/web/t05st10",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        payload = {
            "encodeURIComponent": "1", "step": "1", "firstin": "1",
            "off": "1", "TYPEK": "otc", "isnew": "false",
            "year": str(roc_year), "month": f"{month:02d}",
        }
        r2   = requests.post(url, data=payload, headers=hdrs,
                             timeout=25, verify=False)
        html = r2.content.decode("utf-8", errors="ignore")
        import re as _re
        rows = _re.findall(r'<tr[^>]*>(.*?)</tr>', html, _re.DOTALL)
        for row in rows:
            cells = _re.findall(r'<td[^>]*>(.*?)</td>', row, _re.DOTALL)
            cells = [_re.sub(r'<[^>]+>', '', c).strip() for c in cells]
            if len(cells) >= 10 and len(cells[0]) == 4 and cells[0].isdigit():
                try:
                    result[cells[0]] = float(cells[9].replace(",", ""))
                except Exception:
                    pass
    except Exception as e:
        print(f"  月營收上櫃抓取失敗：{e}")

    listed = sum(1 for c in result if not c.startswith("9"))
    print(f"  月營收資料：{len(result)} 支（上市約 {listed}）")
    return result

def fetch_sector_map():
    try:
        import twstock
        smap = {
            code: getattr(info, "group", "—")
            for code, info in twstock.codes.items()
            if getattr(info, "market", "") in ("上市", "上櫃") and getattr(info, "group", "")
        }
        if smap:
            return smap
    except Exception:
        pass
    try:
        _, smap = _parse_isin_page(2)
        return smap
    except Exception:
        return {}

# ── 技術指標 ──────────────────────────────────────────────────────────────────
def calc_indicators(df, fast_ma, slow_ma):
    df = df.copy()
    for ma in [5, 10, 20, 60]:
        df[f"MA{ma}"] = df["Close"].rolling(ma).mean()
    fast_col = f"MA{fast_ma}"
    slow_col = f"MA{slow_ma}"
    df["gap"]      = df[fast_col] - df[slow_col]
    df["gap_prev"] = df["gap"].shift(1)
    df["signal"]   = 0
    df.loc[(df["gap"] > 0) & (df["gap_prev"] <= 0), "signal"] = 1
    df.loc[(df["gap"] < 0) & (df["gap_prev"] >= 0), "signal"] = -1
    direction    = np.sign(df["Close"].diff().fillna(0))
    df["OBV"]    = (direction * df["Volume"]).cumsum()
    for n, label in [(21, "R1M"), (63, "R3M"), (126, "R6M")]:
        df[label] = (df["Close"] / df["Close"].shift(n) - 1) * 100
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI14"] = 100 - 100 / (1 + gain / (loss + 1e-9))
    _hl = df["High"] - df["Low"]
    _hc = (df["High"] - df["Close"].shift()).abs()
    _lc = (df["Low"]  - df["Close"].shift()).abs()
    _atr_s = pd.concat([_hl, _hc, _lc], axis=1).max(axis=1).rolling(14).mean() + 1e-9
    _hdiff = df["High"].diff()
    _ldiff = -df["Low"].diff()
    _plus_dm  = _hdiff.where((_hdiff > _ldiff) & (_hdiff > 0), 0.0)
    _minus_dm = _ldiff.where((_ldiff > _hdiff) & (_ldiff > 0), 0.0)
    _plus_di  = 100 * (_plus_dm.rolling(14).mean() / _atr_s)
    _minus_di = 100 * (_minus_dm.rolling(14).mean() / _atr_s)
    _dx = 100 * (_plus_di - _minus_di).abs() / (_plus_di + _minus_di + 1e-9)
    df["ADX14"] = _dx.rolling(14).mean()
    return df, fast_col, slow_col

# ── 型態偵測 ──────────────────────────────────────────────────────────────────
def _detect_vcp(c, v):
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
    n = len(c)
    if n < 40:
        return None
    w = 7
    raw_lows = [i for i in range(w, n - w) if c[i] <= min(c[i-w:i+w+1]) + 1e-9]
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

def _detect_divergence(df, lookback=25):
    hits = []
    if len(df) < lookback + 5:
        return hits
    window = df.tail(lookback)
    half   = lookback // 2
    first  = window.iloc[:half]
    second = window.iloc[half:]
    tol_pct = 0.015
    if "OBV" in df.columns:
        ph1, ph2 = first["Close"].max(),  second["Close"].max()
        oh1, oh2 = first["OBV"].max(),    second["OBV"].max()
        pl1, pl2 = first["Close"].min(),  second["Close"].min()
        ol1, ol2 = first["OBV"].min(),    second["OBV"].min()
        if all(pd.notna(x) for x in (ph1, ph2, oh1, oh2)):
            if ph2 > ph1 * (1 + tol_pct) and oh2 < oh1:
                hits.append("OBV頂背離")
        if all(pd.notna(x) for x in (pl1, pl2, ol1, ol2)):
            if pl2 < pl1 * (1 - tol_pct) and ol2 > ol1:
                hits.append("OBV底背離")
    if "RSI14" in df.columns:
        ph1, ph2 = first["Close"].max(),   second["Close"].max()
        rh1, rh2 = first["RSI14"].max(),   second["RSI14"].max()
        pl1, pl2 = first["Close"].min(),   second["Close"].min()
        rl1, rl2 = first["RSI14"].min(),   second["RSI14"].min()
        if all(pd.notna(x) for x in (ph1, ph2, rh1, rh2)):
            if ph2 > ph1 * (1 + tol_pct) and rh2 < rh1 - 3:
                hits.append("RSI頂背離")
        if all(pd.notna(x) for x in (pl1, pl2, rl1, rl2)):
            if pl2 < pl1 * (1 - tol_pct) and rl2 > rl1 + 3:
                hits.append("RSI底背離")
    return hits

def detect_all_patterns(df):
    c = df["Close"].values.astype(float)
    v = df["Volume"].values.astype(float)
    hits = []
    for fn in (_detect_vcp, _detect_flat_base):
        r = fn(c, v)
        if r:
            hits.append(r)
    r = _detect_double_bottom(c)
    if r:
        hits.append(r)
    r = _detect_cup_handle(c, v)
    if r:
        hits.append(r)
    hits.extend(_detect_divergence(df))
    return hits

# ── 相對強度 vs 0050 ────────────────────────────────────────────────────────
def _fetch_benchmark():
    try:
        df = yf.download("0050.TW", period=PERIOD, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        return df if not df.empty else None
    except Exception:
        return None

def _calc_rs(df, bench_df, n=63):
    try:
        if bench_df is None or len(df) <= n or len(bench_df) <= n:
            return None
        s_ret = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-1-n]) - 1) * 100
        b_ret = (float(bench_df["Close"].iloc[-1]) / float(bench_df["Close"].iloc[-1-n]) - 1) * 100
        return round(s_ret - b_ret, 1)
    except Exception:
        return None

# ── 評分 ──────────────────────────────────────────────────────────────────────
def calc_score(df, patterns, fast_col, slow_col, rs_val=None, revenue_yoy=None):
    latest = df.iloc[-1]
    close  = float(latest["Close"])
    if float(latest[fast_col]) <= float(latest[slow_col]):
        return 0
    top_divs = [p for p in patterns if "頂背離" in p]
    rsi_now  = float(latest["RSI14"]) if "RSI14" in df.columns and pd.notna(latest.get("RSI14")) else 50
    r1m      = float(df["R1M"].iloc[-1]) if "R1M" in df.columns and pd.notna(df["R1M"].iloc[-1]) else 0
    score = 0
    sig_df = df[df["signal"] == 1]
    if not sig_df.empty:
        days_since = (df.index[-1] - sig_df.iloc[-1].name).days
        if days_since <= 2:   score += 35
        elif days_since <= 5: score += 28
        elif days_since <= 10: score += 18
        elif days_since <= 20: score += 10
        elif days_since <= 45: score += 4
    vol_avg = df["Volume"].rolling(20).mean().iloc[-1]
    if pd.notna(vol_avg) and float(vol_avg) > 0:
        vol_ratio = float(latest["Volume"]) / float(vol_avg)
        if vol_ratio >= 2.0:   score += 20
        elif vol_ratio >= 1.5: score += 14
        elif vol_ratio >= 1.0: score += 7
        elif vol_ratio >= 0.7: score += 2
    pos_pats = [p for p in patterns if "頂背離" not in p]
    PAT_W = {"VCP": 12, "杯柄": 10, "雙底": 9, "平台底": 7, "OBV底背離": 8, "RSI底背離": 5}
    score += min(20, sum(w for k, w in PAT_W.items() if any(k in p for p in pos_pats)))
    score -= len(top_divs) * 20
    _breakout_pat = any(k in p for k in ("VCP", "杯柄") for p in pos_pats)
    if 52 <= rsi_now <= 65:   score += 10
    elif (45 <= rsi_now < 52) or (65 < rsi_now <= 72): score += 5
    elif rsi_now > 75 and not _breakout_pat: score -= 10
    if 3 < r1m <= 15:         score += 10
    elif 0 < r1m <= 3 or 15 < r1m <= 25: score += 5
    elif r1m > 30:            score -= 8
    if rs_val is not None:
        if rs_val >= 15:      score += 10
        elif rs_val >= 5:     score += 6
        elif rs_val >= 0:     score += 2
        elif rs_val <= -15:   score -= 12
        elif rs_val <= -5:    score -= 6
    high_52 = df["High"].tail(252).max() if len(df) >= 150 else df["High"].max()
    if float(high_52) > 0:
        dist_pct = (close / float(high_52) - 1) * 100
        if -8 <= dist_pct <= 0:     score += 5
        elif -20 <= dist_pct < -8:  score += 3
        elif dist_pct < -40:        score -= 3
    if "MA60" in df.columns and pd.notna(df["MA60"].iloc[-1]):
        if pd.notna(df[slow_col].iloc[-1]) and float(df[slow_col].iloc[-1]) > float(df["MA60"].iloc[-1]):
            score += 5
    if revenue_yoy is not None:
        if revenue_yoy >= 30:    score += 15
        elif revenue_yoy >= 15:  score += 8
        elif revenue_yoy >= 5:   score += 3
        elif revenue_yoy <= -20: score -= 8
    cap = 40 if (top_divs or (rsi_now >= 78 and not _breakout_pat) or r1m > 30) else 100
    return max(0, min(cap, score))

# ── 單支股票分析 ───────────────────────────────────────────────────────────────
def _analyze_one(code, stock_df, bench_df=None, sector="—", revenue_yoy=None):
    stock_df = stock_df.copy()
    stock_df.index = pd.to_datetime(stock_df.index)
    stock_df = stock_df.dropna(subset=["Close"])
    if len(stock_df) < max(SLOW_MA + 10, 40):
        return None
    df, fast_col, slow_col = calc_indicators(stock_df, FAST_MA, SLOW_MA)
    avg_vol = df["Volume"].tail(20).mean() if "Volume" in df.columns else 0
    latest_close = float(df["Close"].iloc[-1]) if not df.empty else 0
    if avg_vol < 500 or (avg_vol * latest_close) < 5_000_000:
        return None
    latest  = df.iloc[-1]
    close   = float(latest["Close"])
    gap_now = float(latest[fast_col]) - float(latest[slow_col])
    trend   = "多頭" if gap_now > 0 else "空頭"
    sig_df  = df[df["signal"] != 0]
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
    _r1m = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-21]) - 1) * 100 if len(df) > 21 else 0
    _r3m = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-63]) - 1) * 100 if len(df) > 63 else 0
    is_extended = (_r1m > 25 and not consol_pat) or (_r3m > 60 and not consol_pat)
    _trigger = _extract_trigger_price(patterns)
    broke_out = (_trigger is not None and _trigger * 0.98 <= close <= _trigger * 1.08)

    if top_divs:
        action = "不看"
    elif is_extended:
        action = "不看"
    elif consol_pat and broke_out and trend == "多頭":
        action = "可買"
    elif consol_pat and trend == "多頭":
        action = "等待進場"
    elif trend == "多頭" and tag in ("近期黃金交叉", "多頭持續中"):
        action = "等待進場"
    else:
        action = "不看"
    rs_val  = _calc_rs(df, bench_df)
    rsi_cur = float(df["RSI14"].iloc[-1]) if "RSI14" in df.columns and pd.notna(df["RSI14"].iloc[-1]) else 50
    if rs_val is not None and rs_val < -15:
        action = "不看"
    elif rsi_cur > 73 and _r1m > 18 and action == "等待進場":
        action = "不看"
    # 收盤 < MA5 → 假突破，降為等待進場
    if action == "可買":
        _ma5 = float(df["MA5"].iloc[-1]) if "MA5" in df.columns and pd.notna(df["MA5"].iloc[-1]) else 0
        if _ma5 > 0 and close < _ma5:
            action = "等待進場"
    # ADX < 20 → 盤整市，均線交叉不可信
    if action == "可買":
        _adx = float(df["ADX14"].iloc[-1]) if "ADX14" in df.columns and pd.notna(df["ADX14"].iloc[-1]) else 0
        if _adx < 20:
            action = "等待進場"
    # 多訊號確認：黃金交叉需搭配至少 1 個輔助訊號
    if action == "可買":
        _vol_avg = df["Volume"].tail(20).mean()
        _vol_ratio = float(latest["Volume"]) / float(_vol_avg) if pd.notna(_vol_avg) and float(_vol_avg) > 0 else 0
        _rsi_5d_min = float(df["RSI14"].tail(5).min()) if "RSI14" in df.columns else 50
        _aux = 0
        if _vol_ratio >= 1.5:                                    _aux += 1
        if consol_pat:                                           _aux += 1
        if _rsi_5d_min <= 40 and rsi_cur > 40:                  _aux += 1
        if _aux < 1:
            action = "等待進場"

    _CYCLICAL = {"建材營造業", "水泥工業"}
    if action == "可買" and sector in _CYCLICAL:
        if revenue_yoy is None or revenue_yoy <= 0:
            action = "等待進場"

    score   = calc_score(df, patterns, fast_col, slow_col, rs_val, revenue_yoy)
    high_52 = df["High"].tail(252).max() if len(df) >= 150 else df["High"].max()
    dist_52h = round((close / high_52 - 1) * 100, 1) if high_52 > 0 else None
    rsi_now  = round(float(df["RSI14"].iloc[-1]), 1) if "RSI14" in df.columns and pd.notna(df["RSI14"].iloc[-1]) else None
    r1m_val  = round(float(df["R1M"].iloc[-1]), 1) if "R1M" in df.columns and pd.notna(df["R1M"].iloc[-1]) else None
    trigger  = _extract_trigger_price(patterns)
    return {
        "分數": score,
        "代號": code,
        "產業": sector,
        "操作": action,
        "收盤價": round(close, 2),
        "RSI": rsi_now if rsi_now is not None else "—",
        "1M%": f"{r1m_val:+.1f}%" if r1m_val is not None else "—",
        "距52高%": f"{dist_52h:+.1f}%" if dist_52h is not None else "—",
        "RS大盤": f"{rs_val:+.1f}%" if rs_val is not None else "—",
        "狀態": tag,
        "訊號日": last_date,
        "幾天前": days_ago if days_ago < 999 else "—",
        "型態": "、".join(patterns) if patterns else "—",
        "月營收YoY": f"{revenue_yoy:+.0f}%" if revenue_yoy is not None else "—",
        "觸發價": f"{trigger:.1f}" if trigger is not None else "—",
    }

# ── MOPS 重大公告監控 ─────────────────────────────────────────────────────────
def fetch_mops_news():
    """抓取 MOPS 今日重大訊息，過濾出催化劑型公告"""
    date_str = datetime.today().strftime("%Y%m%d")
    url = "https://mops.twse.com.tw/mops/web/ajax_t05sr01_1"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://mops.twse.com.tw/mops/web/t05sr01_1",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    payload = {
        "encodeURIComponent": "1", "step": "1", "firstin": "1",
        "off": "1", "keyword4": "", "code1": "", "TYPEK2": "",
        "checkbtn": "", "queryName": "co_id", "inpuData": "",
        "co_id": "", "begin_date": date_str, "end_date": date_str,
    }
    results = []
    try:
        r = requests.post(url, data=payload, headers=headers, timeout=25)
        html = r.content.decode("utf-8", errors="ignore")
        # 擷取每列 td 內容
        raw_rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL)
        for row in raw_rows:
            cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
            cells = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
            if len(cells) < 5:
                continue
            code = cells[0]
            if not (len(code) == 4 and code.isdigit()):
                continue
            subject = cells[4] if len(cells) > 4 else ""
            if not any(kw in subject for kw in MOPS_KEYWORDS):
                continue
            results.append({
                "代號": code,
                "公司": cells[1] if len(cells) > 1 else "",
                "時間": cells[3] if len(cells) > 3 else "",
                "主旨": subject,
            })
    except Exception as e:
        print(f"  MOPS 抓取失敗：{e}")
    return results

# ── 批次下載 + 掃描 ────────────────────────────────────────────────────────────
def _fetch_multi(symbols_tuple):
    syms = list(symbols_tuple)
    def _dl():
        return yf.download(syms, period=PERIOD, auto_adjust=True,
                           progress=False, group_by="ticker", threads=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        try:
            return ex.submit(_dl).result(timeout=60)
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

def scan_stocks(codes, bench_df, sector_map, suffix=".TW", revenue_dict=None):
    results = []
    total = len(codes)
    for i in range(0, total, 100):
        batch_codes = codes[i: i + 100]
        batch_syms  = [f"{c}{suffix}" for c in batch_codes]
        print(f"\r  {i+1}–{min(i+100, total)} / {total}", end="", flush=True)
        try:
            multi = _fetch_multi(tuple(batch_syms))
        except Exception:
            continue
        for code, sym in zip(batch_codes, batch_syms):
            try:
                sdf = _extract_one(multi, sym, len(batch_syms))
                if sdf is None:
                    continue
                row = _analyze_one(code, sdf, bench_df=bench_df,
                                   sector=sector_map.get(code, "—"),
                                   revenue_yoy=(revenue_dict or {}).get(code))
                if row:
                    results.append(row)
            except Exception:
                continue
    print()
    return pd.DataFrame(results)

# ── 寄信 ─────────────────────────────────────────────────────────────────────
def _build_table_html(df, header_color="#2c3e50"):
    cols = ["分數", "代號", "產業", "操作", "收盤價", "RSI", "1M%",
            "距52高%", "RS大盤", "月營收YoY", "連買天", "外資累計(張)", "幾天前", "連掃天", "觸發價", "型態"]
    cols = [c for c in cols if c in df.columns]
    def _row_color(op):
        return "#d4edda" if op == "可買" else "#fff3cd" if op == "等待進場" else "white"
    rows_html = ""
    for _, r in df.iterrows():
        bg    = _row_color(r.get("操作", ""))
        cells = "".join(f"<td style='padding:5px 8px;border-bottom:1px solid #ddd'>{r.get(c,'')}</td>" for c in cols)
        rows_html += f"<tr style='background:{bg}'>{cells}</tr>"
    headers = "".join(
        f"<th style='padding:6px 8px;text-align:left;background:{header_color};color:white'>{c}</th>"
        for c in cols
    )
    return f"""<table style='border-collapse:collapse;width:100%;font-size:13px'>
<thead><tr>{headers}</tr></thead>
<tbody>{rows_html}</tbody>
</table>"""


def send_email(df_tw, df_otc, total_scanned, mops_news=None):
    _days = ["一", "二", "三", "四", "五", "六", "日"]
    today  = datetime.now().strftime("%Y-%m-%d") + f"（週{_days[datetime.now().weekday()]}）"
    n_tw   = len(df_tw)
    n_otc  = len(df_otc)

    # ── 上市區塊 ──
    if df_tw.empty:
        tw_html = f"<p>今日無上市股票符合條件（分數 ≥ {MIN_SCORE}）。</p>"
    else:
        tw_html = _build_table_html(df_tw, header_color="#2c3e50")

    # ── 上櫃區塊 ──
    if df_otc.empty:
        otc_html = f"<p>今日無上櫃股票符合條件（分數 ≥ {MIN_SCORE_OTC}）。</p>"
    else:
        otc_html = _build_table_html(df_otc, header_color="#1a6b3c")

    # ── MOPS 重大公告 ──
    mops_html = ""
    if mops_news:
        mops_rows = "".join(
            f"<tr>"
            f"<td style='padding:4px 8px;border-bottom:1px solid #eee'>{m['代號']}</td>"
            f"<td style='padding:4px 8px;border-bottom:1px solid #eee'>{m['公司']}</td>"
            f"<td style='padding:4px 8px;border-bottom:1px solid #eee'>{m['時間']}</td>"
            f"<td style='padding:4px 8px;border-bottom:1px solid #eee'>{m['主旨']}</td>"
            f"</tr>"
            for m in mops_news
        )
        mops_html = f"""
<h3 style='color:#c0392b;margin-top:30px'>今日重大公告（{len(mops_news)} 筆）</h3>
<p style='color:#666;font-size:12px'>來源：MOPS 公開資訊觀測站｜含關鍵字：資產處分、重大合約、取得重大資產</p>
<table style='border-collapse:collapse;width:100%;font-size:13px'>
<thead><tr>
  <th style='padding:5px 8px;background:#c0392b;color:white'>代號</th>
  <th style='padding:5px 8px;background:#c0392b;color:white'>公司</th>
  <th style='padding:5px 8px;background:#c0392b;color:white'>時間</th>
  <th style='padding:5px 8px;background:#c0392b;color:white'>主旨</th>
</tr></thead>
<tbody>{mops_rows}</tbody>
</table>"""

    html = f"""<html><body style='font-family:Arial,sans-serif;padding:20px'>
<h2 style='color:#2c3e50'>台股每日掃描 — {today}</h2>
<p>掃描 {total_scanned} 支股票 | 上市 {n_tw} 支（分數≥{MIN_SCORE}，依連買天排序）/ 上櫃 {n_otc} 支（分數≥{MIN_SCORE_OTC}，依分數排序）</p>
<h3 style='color:#2c3e50;margin-top:20px'>上市（TWSE）— {n_tw} 支</h3>
{tw_html}
<h3 style='color:#1a6b3c;margin-top:30px'>上櫃（TPEX）— {n_otc} 支</h3>
{otc_html}
{mops_html}
<p style='color:#999;font-size:11px;margin-top:20px'>
  均線 MA{FAST_MA}/MA{SLOW_MA} · 止損 {STOP_PCT}% · 期間 {PERIOD} · 綠=可買 黃=等待進場
</p>
</body></html>"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"【台股掃描】{datetime.now().strftime('%m/%d')} — 上市{n_tw}支 上櫃{n_otc}支"
    msg["From"]    = GMAIL_USER
    msg["To"]      = TO_EMAIL
    msg.attach(MIMEText(html, "html", "utf-8"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(GMAIL_USER, GMAIL_APP_PW)
        s.sendmail(GMAIL_USER, TO_EMAIL, msg.as_string())

# ── 主程式 ────────────────────────────────────────────────────────────────────
def main():
    print(f"[{datetime.now():%H:%M:%S}] 台股每日掃描啟動")

    if not GMAIL_APP_PW:
        print("錯誤：請設定環境變數 GMAIL_APP_PW（Gmail App Password）")
        sys.exit(1)

    print("抓取 T86 法人資料...")
    inst_data = _fetch_institutional_data()
    print(f"  外資有買超：{len(inst_data)} 支")

    print("抓取股票代號...")
    codes_tw  = fetch_twse_codes()
    codes_two = fetch_otc_codes() if SCAN_OTC else []
    print(f"  上市：{len(codes_tw)} 支 ／ 上櫃：{len(codes_two)} 支")

    print("抓取產業/基準/月營收資料...")
    sector_map   = fetch_sector_map()
    bench_df     = _fetch_benchmark()
    revenue_dict = fetch_monthly_revenue()

    print("開始掃描上市（每批 100 支）：")
    result_tw = scan_stocks(codes_tw, bench_df, sector_map, suffix=".TW",
                            revenue_dict=revenue_dict)
    print(f"  上市有效：{len(result_tw)} 支")

    result_two = pd.DataFrame()
    if SCAN_OTC and codes_two:
        print("開始掃描上櫃（每批 100 支）：")
        result_two = scan_stocks(codes_two, bench_df, sector_map, suffix=".TWO",
                                 revenue_dict=revenue_dict)
        print(f"  上櫃有效：{len(result_two)} 支")

    # 上市加法人籌碼
    if not result_tw.empty:
        result_tw["連買天"]       = result_tw["代號"].map(lambda c: inst_data.get(c, {}).get("consec", 0))
        result_tw["外資累計(張)"] = result_tw["代號"].map(lambda c: inst_data.get(c, {}).get("total", 0) // 1000)

    # 上櫃無 T86，填 0
    if not result_two.empty:
        result_two["連買天"]       = 0
        result_two["外資累計(張)"] = 0

    # 上市：依連買天→分數排序，TOP N
    filtered_tw = pd.DataFrame()
    if not result_tw.empty:
        filtered_tw = result_tw[
            (result_tw["分數"] >= MIN_SCORE) &
            (result_tw["操作"].isin(["可買", "等待進場"]))
        ].sort_values(["連買天", "分數"], ascending=[False, False]).head(TOP_N).copy()
        print(f"  上市符合（分數≥{MIN_SCORE}）：{len(filtered_tw)} 支")

    # 上櫃：獨立門檻，依分數排序，TOP N
    filtered_otc = pd.DataFrame()
    if not result_two.empty:
        filtered_otc = result_two[
            (result_two["分數"] >= MIN_SCORE_OTC) &
            (result_two["操作"].isin(["可買", "等待進場"]))
        ].sort_values("分數", ascending=False).head(TOP_N).copy()
        print(f"  上櫃符合（分數≥{MIN_SCORE_OTC}）：{len(filtered_otc)} 支")

    today_str = datetime.now().strftime("%Y-%m-%d")
    hist = _load_scan_history()
    for df_ in [filtered_tw, filtered_otc]:
        if not df_.empty:
            df_["連掃天"] = df_["代號"].apply(lambda c: _count_streak(c, hist, today_str))
    all_codes = (filtered_tw["代號"].tolist() if not filtered_tw.empty else []) + \
                (filtered_otc["代號"].tolist() if not filtered_otc.empty else [])
    _save_scan_history(today_str, all_codes)

    print("抓取 MOPS 重大公告...")
    mops_news = fetch_mops_news()
    print(f"  重大公告：{len(mops_news)} 筆")

    print("寄送郵件...")
    total_scanned = len(codes_tw) + len(codes_two)
    send_email(filtered_tw, filtered_otc, total_scanned, mops_news=mops_news)
    print(f"[{datetime.now():%H:%M:%S}] 完成")

if __name__ == "__main__":
    main()
