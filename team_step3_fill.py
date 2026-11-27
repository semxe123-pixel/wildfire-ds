"""공통 학습 표 Step 3: 입력 24열의 빈 값 채우기 → data/gold/team/filled.parquet, data/gold/team/fill_log.csv

채우는 순서 (앞 방법으로 못 채운 칸만 다음 방법으로)
  1 실제 관측   (a) 2024-01 초 산불: 2023-12 기상이 우리 자료에 없음 → 완길님 Silver의 2023-12 값으로 우리 규칙대로 다시 계산
                (b) 돌풍: 완길님이 본 관측소 3곳에 값 없음 → 우리 후보 관측소(풍속 1–5순위)의 돌풍 값을 완길님 Silver에서 찾음
  2 물리적으로 정해지는 값   wind_u·wind_v: 풍속 0.2 m/s 이하면 바람 벡터 = 0
  3 다른 열로 추정   돌풍 ← 풍속·24시간 최대 풍속 (학습 구간에서 배운 선형 관계)
  4 학습 구간 중앙값
학습 구간 = 발생 2025-03-14 이전 (Step 4의 train과 같음). 3·4는 학습 구간 값만 써서 계산 (미래 정보 없음)
실행: python team_step3_fill.py   (wildfire 폴더에서, Step 2 다음)
"""
import glob
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
TZ = "Asia/Seoul"
IN, OUT, LOG = "data/gold/team/joined.parquet", "data/gold/team/filled.parquet", "data/gold/team/fill_log.csv"
ASSIGN = "data/silver/stn_assign.parquet"
OUR_AWS = "data/silver/aws_silver_{}.parquet"
TEAM_AWS = "../wildfire_team/제출실버 3조/data/silver/weather/aws_hourly/year={}/month={}/*.parquet"
DATA_START = pd.Timestamp("2024-01-01", tz=TZ)
VALID_START = pd.Timestamp("2025-03-15", tz=TZ)
FEATURES = ["ta", "hm", "ws", "wind_u", "wind_v", "pa", "ps", "gust_max_1h",
            "hm_min_6h", "hm_min_24h", "hm_mean_24h", "eff_hm", "ta_max_24h", "ta_min_24h",
            "ws_mean_6h", "ws_max_24h", "gust_max_6h", "rn_sum_72h", "rn_sum_7d", "dry_hours",
            "hour", "month", "dayofweek", "is_weekend"]
WINDOW = ["ta_max_24h", "ta_min_24h", "hm_min_24h", "hm_mean_24h", "ws_max_24h", "ws_mean_6h",
          "rn_sum_72h", "dry_hours", "eff_hm"]
OUR_COL = {"TA": "TA", "HM": "HM", "WS": "WS", "RN_HR1": "RN_HR1"}
TEAM_COL = {"TA": "temperature_instant_c", "HM": "relative_humidity_instant_pct",       # 우리 정시값과 같은 정의
            "WS": "wind_speed_10min_ms", "RN_HR1": "rainfall_1h_mm", "GUST": "wind_gust_max_1h_ms"}
LB, COVER, R_EFF = 120, 0.75, 0.7                                                        # Gold F5·F6·F7과 같음

df = pd.read_parquet(IN)
df["ref"] = pd.to_datetime(df.forecast_reference_time).dt.tz_convert(TZ)
df["is_train"] = pd.to_datetime(df.ignition_at).dt.tz_convert(TZ) < VALID_START
before = df[FEATURES].copy()
log = []


def note(i, col, new, method):
    log.append({"fire_id": df.at[i, "fire_id"], "column": col, "before": before.at[i, col], "after": new, "method": method})
    df.at[i, col] = new


# ---------------------------------------------------------------- 자료 읽기 (빈 값이 있는 산불에 필요한 달만)
need = df[df[FEATURES].isna().any(axis=1)]
win_rows = df[df[WINDOW].isna().any(axis=1) & (df.ref - pd.Timedelta(hours=LB) < DATA_START)]     # 1(a) 대상
gust_rows = df[df[["gust_max_1h", "gust_max_6h"]].isna().any(axis=1)]                               # 1(b) 대상
spans = [(r - pd.Timedelta(hours=LB + 1), r) for r in win_rows.ref] + [(r - pd.Timedelta(hours=7), r) for r in gust_rows.ref]
months = sorted({p.strftime("%Y%m") for a, b in spans for p in pd.date_range(a, b, freq="h")})
assign = pd.read_parquet(ASSIGN)
our = pd.concat([pd.read_parquet(OUR_AWS.format(m)) for m in months if pd.Timestamp(m[:4] + "-" + m[4:] + "-01", tz=TZ) >= DATA_START],
                ignore_index=True)
our_good = {e: our[our[f"qc_{c}"].isin([0, 2]) & our[c].notna()].set_index(["station_id", "observed_at"])[c]
            for e, c in OUR_COL.items()}                                                 # Gold F2와 같음: 판정 0·2만
team = pd.concat([pd.read_parquet(f) for m in months for f in glob.glob(TEAM_AWS.format(int(m[:4]), int(m[4:])))],
                 ignore_index=True).drop_duplicates(["station_id", "observed_at"])
team_val = {e: team[team[c].notna()].set_index(["station_id", "observed_at"])[c] for e, c in TEAM_COL.items()}
print(f"빈 값이 있는 산불 {len(need)}건 · 읽은 달 {months}")


def candidates(row, element):
    """그 산불의 행정동·기준 시각에 유효한 후보 관측소 (가까운 순)"""
    a = assign[(assign.admin_dong_code == row.admin_dong_code) & (assign.month == row.ref.strftime("%Y%m"))
               & (assign.element == element) & (assign.valid_from <= row.ref) & (assign.valid_to > row.ref)]
    return a.sort_values("rank").station_id.tolist()


def series(row, element, hours, source):
    """시각마다 값이 있는 첫 후보의 값 (Gold F2). source: 'mixed' = 2024 이후는 우리 Silver, 이전은 완길님 Silver"""
    st = candidates(row, "WS" if element == "GUST" else element)
    out = []
    for h in hours:
        tab = team_val[element] if (source == "team" or h < DATA_START) else our_good[element]
        out.append(next((float(tab[(s, h)]) for s in st if (s, h) in tab.index), np.nan))
    return np.array(out)


def stat(v, how, w):
    """마지막 w시간 통계. 값이 75% 미만이면 빈 값 (Gold F5)"""
    x = v[-w:]
    if np.isfinite(x).sum() < COVER * w:
        return np.nan
    return {"max": np.nanmax, "min": np.nanmin, "mean": np.nanmean, "sum": np.nansum}[how](x)


# ---------------------------------------------------------------- 1(a) 2024-01 초: 2023-12 자료로 누적 값 다시 계산
for i, row in df[df[WINDOW].isna().any(axis=1)].iterrows():
    hours = pd.date_range(row.ref - pd.Timedelta(hours=LB), row.ref - pd.Timedelta(hours=1), freq="h")   # 기준 시각 - 1시간까지 (F1)
    if hours[0] >= DATA_START:
        continue                                                                         # 자료 부족 때문이 아님 → 뒤 방법으로
    S = {e: series(row, e, hours, "mixed") for e in ("TA", "HM", "WS", "RN_HR1")}
    dry, d = np.nan, 0
    for r in S["RN_HR1"]:
        d = 0 if r >= 0.1 else (d if np.isnan(r) else min(d + 1, LB))
        dry = d
    he = [stat(S["HM"][: len(hours) - 24 * k], "mean", 24) for k in range(5)]
    calc = {"ta_max_24h": stat(S["TA"], "max", 24), "ta_min_24h": stat(S["TA"], "min", 24),
            "hm_min_24h": stat(S["HM"], "min", 24), "hm_mean_24h": stat(S["HM"], "mean", 24),
            "ws_max_24h": stat(S["WS"], "max", 24), "ws_mean_6h": stat(S["WS"], "mean", 6),
            "rn_sum_72h": stat(S["RN_HR1"], "sum", 72), "dry_hours": float(dry),
            "eff_hm": (1 - R_EFF) * sum(R_EFF ** k * h for k, h in enumerate(he))}
    for c in WINDOW:
        if pd.isna(df.at[i, c]) and np.isfinite(calc[c]):
            note(i, c, round(float(calc[c]), 3), "1 실제 관측: 2023-12 완길님 Silver + 우리 Silver로 다시 계산")

# ---------------------------------------------------------------- 1(b) 돌풍: 더 먼 후보 관측소의 실제 값
for i, row in df[df[["gust_max_1h", "gust_max_6h"]].isna().any(axis=1)].iterrows():
    hours = pd.date_range(row.ref - pd.Timedelta(hours=6), row.ref - pd.Timedelta(hours=1), freq="h")
    g = series(row, "GUST", hours, "team")
    if pd.isna(df.at[i, "gust_max_1h"]) and np.isfinite(g[-1]):
        note(i, "gust_max_1h", float(g[-1]), "1 실제 관측: 우리 후보 관측소(풍속 1–5순위)의 돌풍")
    if pd.isna(df.at[i, "gust_max_6h"]) and np.isfinite(g).sum() >= 3:                  # 완길님 규칙: 창 안 50% 이상
        note(i, "gust_max_6h", float(np.nanmax(g)), "1 실제 관측: 우리 후보 관측소(풍속 1–5순위)의 돌풍")

# ---------------------------------------------------------------- 2 바람 벡터: 풍속 0.2 m/s 이하면 0
for c in ("wind_u", "wind_v"):
    for i in df.index[df[c].isna() & (df.ws <= 0.2)]:
        note(i, c, 0.0, "2 물리: 풍속 0.2 m/s 이하라 벡터 크기 0.2 이하 → 0")

# ---------------------------------------------------------------- 3 돌풍을 풍속으로 추정 (학습 구간에서 배운 관계)
for c in ("gust_max_1h", "gust_max_6h"):
    tr = df[df.is_train & df[[c, "ws", "ws_max_24h"]].notna().all(axis=1)]
    X = np.c_[np.ones(len(tr)), tr.ws, tr.ws_max_24h]
    coef = np.linalg.lstsq(X, tr[c].to_numpy(float), rcond=None)[0]
    for i in df.index[df[c].isna() & df[["ws", "ws_max_24h"]].notna().all(axis=1)]:
        note(i, c, round(float(coef @ [1, df.at[i, "ws"], df.at[i, "ws_max_24h"]]), 2), "3 추정: 풍속·24시간 최대 풍속 선형 관계")

# ---------------------------------------------------------------- 4 그래도 남으면 학습 구간 중앙값
for c in FEATURES:
    med = float(df.loc[df.is_train, c].median())
    for i in df.index[df[c].isna()]:
        note(i, c, med, "4 학습 구간 중앙값")

# ---------------------------------------------------------------- 확인과 저장
lg = pd.DataFrame(log)
assert df[FEATURES].notna().all().all(), "아직 빈 값이 있음"
print(f"채운 칸 {len(lg)}개 / 산불 {lg.fire_id.nunique()}건")
print(lg.groupby("method").size().to_string())
print(lg.to_string(index=False))
df.drop(columns=["ref", "is_train"]).to_parquet(OUT, index=False)
lg.to_csv(LOG, index=False, encoding="utf-8-sig")
print(f"저장: {OUT}, {LOG}")