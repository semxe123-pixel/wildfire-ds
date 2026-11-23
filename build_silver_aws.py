"""Silver 5단계: 기상(AWS) 매시 자료에 품질검사를 적용해 Silver 표를 만든다.

입력  data/bronze/aws_raw/aws_raw_YYYYMM.parquet (24개월), data/bronze/stn_raw.parquet (관측소 좌표·높이)
출력  data/silver/aws_silver_YYYYMM.parquet   관측소 × 시각 한 행. 값 + 요소별 판정(qc_*) + 걸린 규칙(qc_*_flags)
      data/silver/aws_availability.parquet/.csv 관측소 × 월 × 요소별 값 개수와 판정 개수
      data/silver/aws_qc_summary.csv             규칙별 건수
      data/silver/aws_units.csv                  요소별 이름·단위·설명

판정 (qc_요소)
  0 정상 / 1 결측(기상청 −99) / 2 확인 필요(값 유지, 표시만) / 3 제거(값을 빈 값으로)
걸린 규칙 (qc_요소_flags): 규칙 번호를 쉼표로 이음. 예: "3-2,3-3". 걸린 규칙이 없으면 빈 값

원칙
  - 실시간: 시각 t의 판정에는 t까지의 값(자기 관측소의 과거 값, 같은 시각까지의 이웃 값)만 씀
  - 행은 지우지 않음. 제거는 그 칸의 값만 비움. 원래 값은 Bronze에 남음
  - 빈 값을 채우지 않음 (대체는 Gold에서)
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)       # 이웃이 모두 빈 값인 행의 중앙값 경고
sys.stdout.reconfigure(encoding="utf-8")                          # 파일로 저장할 때 한글이 깨지지 않게
TZ = "Asia/Seoul"
IN_DIR, OUT_DIR = "data/bronze/aws_raw", "data/silver"
NUM = ["TA", "WD", "WS", "RN_DAY", "RN_HR1", "HM", "PA", "PS"]
MAX_KM, K = 30, 3                       # 이웃: 30 km 안에서 가까운 최대 3곳의 중앙값
LAPSE = 0.0065                          # 기온 높이 보정: 100 m 높을 때마다 0.65℃ 낮음
NORMAL, MISSING, CHECK, REMOVE = 0, 1, 2, 3
os.makedirs(OUT_DIR, exist_ok=True)

# ================================================================ 1. 읽기
frames = []
for f in sorted(os.listdir(IN_DIR)):
    if f.endswith(".parquet"):
        frames.append(pd.read_parquet(os.path.join(IN_DIR, f)))
raw = pd.concat(frames, ignore_index=True)
df = pd.DataFrame({
    "observed_at": pd.to_datetime(raw["YYMMDDHHMI"], format="%Y%m%d%H%M").dt.tz_localize(TZ),
    "station_id": raw["STN"].str.strip(),
    "bronze_file": "aws/" + raw["src_file"],
})
df["STN"] = df.station_id.astype("int32")                         # 계산용 숫자 번호 (저장하지 않음)
for c in NUM:
    v = pd.to_numeric(raw[c]).astype("float32")
    df[c] = v.where(v > -99)                                        # 기상청 결측 표시(−99) → 빈 값
del raw, frames
df = df.sort_values(["STN", "observed_at"]).reset_index(drop=True)
df["month"] = df.observed_at.dt.strftime("%Y%m")
dup = int(df.duplicated(["STN", "observed_at"]).sum())
assert dup == 0, f"같은 관측소·시각이 {dup}행 중복됨"
print(f"읽기: {len(df):,}행, 관측소 {df.STN.nunique()}곳, {df.observed_at.min()} ~ {df.observed_at.max()}")

# ================================================================ 2. 관측소 정보 (관측소 × 월의 좌표·높이)
stn = pd.read_parquet("data/bronze/stn_raw.parquet")
info = pd.DataFrame({
    "STN": stn["STN_ID"].astype("int32"),
    "snap": stn["src_file"].str[8:20],                              # stn_aws_YYYYMMDDHHMI.txt
    "LON": pd.to_numeric(stn["LON"], errors="coerce"),
    "LAT": pd.to_numeric(stn["LAT"], errors="coerce"),
    "HT": pd.to_numeric(stn["HT"], errors="coerce"),
})
months = sorted(df.month.unique())
rows = []
for s, g in info.sort_values("snap").groupby("STN"):
    for mo in months:
        cand = g[g.snap >= mo + "010000"]                           # 그 달 1일 또는 그 뒤의 첫 지점정보
        if len(cand):
            rows.append(cand.iloc[0].to_dict() | {"month": mo})
info = pd.DataFrame(rows).drop(columns="snap")
ht = df[["STN", "month"]].merge(info[["STN", "month", "HT"]], on=["STN", "month"], how="left").HT.to_numpy()
print(f"높이를 찾지 못한 행: {int(np.isnan(ht).sum()):,}")

# ================================================================ 3. 도구
n = len(df)
hour = df.observed_at.dt.hour.to_numpy()
# 바로 앞 행이 같은 관측소의 정확히 1시간 전인가
cont = ((df.STN == df.STN.shift()) & (df.observed_at - df.observed_at.shift() == pd.Timedelta(hours=1))).to_numpy()
cont2 = cont & np.roll(cont, 1)                                    # t−1, t−2 모두 정확히 1시간 간격


def shift(a, k=1):
    """배열을 k칸 아래로 (첫 k칸은 빈 값)"""
    out = np.full(len(a), np.nan, dtype="float64")
    out[k:] = a[:-k]
    return out


def run_so_far(a):
    """t까지 같은 값이 몇 시간째 이어지는가 (t 포함). 미래는 보지 않음"""
    same = cont & (a == shift(a)) & ~np.isnan(a)
    rid = np.cumsum(~same)
    return pd.Series(rid).groupby(rid).cumcount().to_numpy() + 1, rid


times = pd.date_range(df.observed_at.min(), df.observed_at.max(), freq="h")
stns = np.sort(df.STN.unique())
si = pd.Series(np.arange(len(stns)), index=stns)
r_t = pd.Series(np.arange(len(times)), index=times).reindex(df.observed_at).to_numpy()
r_s = si.reindex(df.STN).to_numpy()

NB = {}                                                             # (관측소, 월) → (이웃 열 번호, 높이 차)
for mo, m in info.groupby("month"):
    m = m[m.STN.isin(stns)]
    lat, lon = np.radians(m.LAT.to_numpy()), np.radians(m.LON.to_numpy())
    km = 6371 * np.hypot((lon[:, None] - lon[None, :]) * np.cos(lat[:, None]), lat[:, None] - lat[None, :])
    ids, h = m.STN.to_numpy(), m.HT.to_numpy()
    for i, s in enumerate(ids):
        order = [j for j in np.argsort(km[i])[1:11] if km[i][j] <= MAX_KM]
        NB[(s, mo)] = (si[ids[order]].to_numpy(), h[order] - h[i])


def matrix(a):
    M = np.full((len(times), len(stns)), np.nan, dtype="float32")
    M[r_t, r_s] = a
    return M


def nb_stat(M, rows, mode="level", lapse=0.0, max_dh=None):
    """rows마다 이웃 최대 K곳의 중앙값과 이웃 수.
    level: 같은 시각 값(lapse로 높이 보정) / change: 같은 시각의 1시간 변화량 / max_dh: 높이 차 제한(m)"""
    rows = np.asarray(rows)
    med = np.full(len(rows), np.nan, dtype="float32")
    cnt = np.zeros(len(rows), dtype="int8")
    if len(rows) == 0:
        return med, cnt
    sub = df.iloc[rows]
    for (s, mo), g in sub.groupby(["STN", "month"]):
        if (s, mo) not in NB:
            continue
        cols, dh = NB[(s, mo)]
        if max_dh is not None:
            keep_h = np.abs(dh) <= max_dh
            cols, dh = cols[keep_h], dh[keep_h]
        if len(cols) == 0:
            continue
        t = r_t[g.index]
        if mode == "level":
            vals = M[t][:, cols] + lapse * dh
        else:
            vals = M[t][:, cols] - M[np.maximum(t - 1, 0)][:, cols]
            vals[t == 0] = np.nan
        ok = ~np.isnan(vals)
        keep = ok & (np.cumsum(ok, axis=1) <= K)
        vals = np.where(keep, vals, np.nan)
        pos = np.searchsorted(rows, g.index)
        med[pos] = np.nanmedian(vals, axis=1)
        cnt[pos] = keep.sum(axis=1)
    return med, cnt


def idx(mask):
    return np.flatnonzero(mask)


def mask_of(rows):
    m = np.zeros(n, dtype=bool)
    m[rows] = True
    return m


V = {c: df[c].to_numpy().astype("float64") for c in NUM}           # 원래 값
RULES = []                                                          # (규칙 번호, 요소들, 판정, 행 마스크)


def add(rid, elems, status, mask):
    RULES.append((rid, elems, status, np.asarray(mask, dtype=bool)))


def removed(c):
    """지금까지 '제거'로 판정된 행"""
    m = np.zeros(n, dtype=bool)
    for rid, elems, st, mk in RULES:
        if c in elems and st == REMOVE:
            m |= mk
    return m


# ================================================================ 4. 1묶음 물리한계
TA, HM, WS, WD, PA, PS = V["TA"], V["HM"], V["WS"], V["WD"], V["PA"], V["PS"]
add("1-1", ["TA"], REMOVE, (TA < -40) | (TA > 60))                  # 센서 측정 범위 −40~60℃ 밖
add("1-2", ["TA"], CHECK, (TA > 45) & (TA <= 60))                   # 관측된 적 없지만 잴 수는 있음
add("1-3", ["HM"], REMOVE, (HM < 0) | (HM > 100))                   # 상대습도 정의 범위 밖
add("1-4", ["HM"], REMOVE, HM == 0)                                 # 자연 상태에서 0%는 없음 (18건 모두 00시)
for c in ("RN_HR1", "RN_DAY", "WS"):
    add("1-5", [c], REMOVE, V[c] < 0)                               # 비의 양·바람 세기는 음수 불가
add("1-6", ["WD"], REMOVE, (WD < 0) | (WD > 360))                   # 풍향 정의 범위 밖
add("1-7", ["WS"], CHECK, WS > 60)                                  # 아주 드물지만 불가능하지는 않음
add("1-7", ["PS"], CHECK, (PS < 900) | (PS > 1080))
add("1-8a", ["PA"], REMOVE, (PA < 700) | (PA > 1080))               # 높이와 상관없이 한국 어디서도 불가능
pa_exp = 1013.25 * (1 - 2.25577e-5 * ht) ** 5.25588                 # 표준대기에서 관측소 높이의 기압
add("1-8b", ["PA"], CHECK, (PA >= 700) & (PA <= 1080) & (np.abs(PA - pa_exp) > 50))   # 높이 정보 오류일 수도 있음

CLEAN = {c: np.where(removed(c), np.nan, V[c]) for c in NUM}        # 제거한 값은 아래 비교에서 쓰지 않음

# ================================================================ 5. 3묶음 중 기온 0.0 (단계 규칙보다 먼저: 고착 값은 앞 시각 값으로 쓰지 않기 위함)
ta_run, _ = run_so_far(TA)
M_TA = matrix(CLEAN["TA"])
z = idx((TA == 0) & (ta_run >= 3))
z_nb, _ = nb_stat(M_TA, z, "level", LAPSE)
far = np.abs(z_nb) > 5
z_len = ta_run[z]
add("3-1", ["TA"], CHECK, mask_of(z[far & (z_len <= 5)]))           # 3~5시간째 + 이웃과 5℃ 넘게 차이
add("3-2", ["TA"], REMOVE, mask_of(z[far & (z_len >= 6)]))          # 6시간째 이상 + 이웃과 5℃ 넘게 차이
add("3-3", ["TA"], CHECK, mask_of(z[z_len >= 12]))                  # 12시간째 이상 (이웃과 상관없이)

# ================================================================ 6. 2묶음 단계 (실시간)
zero_flag = mask_of(z[(far & (z_len >= 3)) | (z_len >= 12)])
STEP_BASE = {"TA": np.where(zero_flag, np.nan, CLEAN["TA"]), "HM": CLEAN["HM"], "PA": CLEAN["PA"], "PS": CLEAN["PS"]}
STEP = {"TA": ("2-1", 8), "HM": ("2-2", 50), "PA": ("2-3", 5), "PS": ("2-3", 5)}
for c, (rid, thr) in STEP.items():
    v = STEP_BASE[c]
    d = np.where(cont, v - shift(v), np.nan)
    cand = idx(np.abs(d) > thr)
    nbd, cnt = nb_stat(matrix(v), cand, "change")
    if c == "TA":
        hit = cand[np.abs(d[cand] - nbd) > thr]                     # 내 변화 − 이웃 변화 > 8℃ (이웃 없으면 비교 안 됨)
    else:
        weather = (np.sign(nbd) == np.sign(d[cand])) & (np.abs(nbd) > thr / 2)
        hit = cand[(cnt > 0) & ~weather]                            # 이웃은 같은 방향으로 기준 절반을 넘게 변하지 않음
    F = mask_of(hit)
    back = cont2 & (np.abs(v - shift(v, 2)) <= thr)                 # 지금 값이 t−2와 비슷함 = 원래 수준으로 돌아옴
    returned = F & np.roll(F, 1) & back
    add(rid, [c], CHECK, F & ~returned)

# ================================================================ 7. 3묶음 나머지 지속성·이웃 비교
hm_run, _ = run_so_far(HM)
add("3-4", ["HM"], CHECK, (HM < 95) & (hm_run >= 24))               # 95% 미만 같은 값 24시간째 이상
M_HM = matrix(CLEAN["HM"])
sat = idx((HM >= 95) & (HM < 100) & (hm_run >= 24))
s_nb, _ = nb_stat(M_HM, sat, "level")
add("3-5", ["HM"], CHECK, mask_of(sat[s_nb < 70]))                  # 포화 값 고착인데 이웃은 70% 미만
low = idx(CLEAN["HM"] < 10)
l_nb, _ = nb_stat(M_HM, low, "level", max_dh=300)
add("3-6", ["HM"], CHECK, mask_of(low[(l_nb - HM[low]) > 50]))      # 높이 비슷한 이웃보다 50%p 넘게 낮음

ws_run, ws_rid = run_so_far(WS)
calm_total = pd.Series(ws_run).groupby(ws_rid).transform("max").to_numpy()   # 후보를 고르는 데만 씀 (판정에는 안 씀)
c_rows = idx((WS == 0) & (calm_total >= 24))
w_nb, _ = nb_stat(matrix(CLEAN["WS"]), c_rows, "level")
windy = pd.Series((w_nb >= 2).astype(float))
share = windy.groupby(ws_rid[c_rows]).transform(lambda x: x.rolling(24, min_periods=24).mean()).to_numpy()
add("3-7", ["WS"], CHECK, mask_of(c_rows[(ws_run[c_rows] >= 24) & (share >= 0.75)]))   # 지난 24시간 중 75% 이상 이웃 바람 ≥ 2 m/s

for c in ("PA", "PS"):
    r, _ = run_so_far(V[c])
    add("3-8", [c], REMOVE, ~np.isnan(V[c]) & (r >= 12))           # 기압 같은 값 12시간째 이상

# ================================================================ 8. 4묶음 내적 일치성
RH, RD = CLEAN["RN_HR1"], CLEAN["RN_DAY"]
add("4-1", ["RN_HR1", "RN_DAY"], CHECK, RH > RD + 0.05)             # 1시간 강수 > 일강수
late = (hour == 1) & (RD > RH + 0.5)                                # 01시인데 전날 합계가 남아 있음 (초기화 지연)
add("4-2", ["RN_DAY"], CHECK, late)
dec = cont & (RD < shift(RD) - 0.05)
after_late = (hour == 2) & np.roll(late, 1) & cont                  # 늦게 이루어진 초기화 (02시 값이 맞는 값)
add("4-3", ["RN_DAY"], CHECK, dec & (hour != 1) & ~after_late)
pa_c, ta_c = CLEAN["PA"], CLEAN["TA"]
ps_est = pa_c * (1 - (LAPSE * ht) / (ta_c + LAPSE * ht + 273.15)) ** -5.257   # 현지기압·기온·높이로 계산한 해면기압
mis = np.abs(CLEAN["PS"] - ps_est) > 10
add("4-4", ["PA", "PS"], CHECK, mis)

# ================================================================ 9. 판정 합치기 → 값 비우기
summary = []
year = df.observed_at.dt.year.to_numpy()
for c in NUM:
    has = ~np.isnan(V[c])
    st = np.where(has, NORMAL, MISSING).astype("int8")
    fl = np.full(n, "", dtype=object)
    for rid, elems, s, mk in RULES:
        if c not in elems:
            continue
        hit = mk & has
        st[hit] = np.maximum(st[hit], s)                            # 여러 규칙에 걸리면 제거 > 확인 필요
        h = idx(hit)
        fl[h] = np.where(fl[h] == "", rid, fl[h] + "," + rid)
        yr = year[h]
        summary.append({"rule_id": rid, "element": c, "status": "제거" if s == REMOVE else "확인 필요",
                        "rows": len(h), "stations": df.STN.iloc[h].nunique(),
                        "rows_2024": int((yr == 2024).sum()), "rows_2025": int((yr == 2025).sum())})
    df[f"qc_{c}"] = st
    df[f"qc_{c}_flags"] = pd.array(np.where(fl == "", pd.NA, fl), dtype="string")
    df[c] = df[c].where(st != REMOVE)                               # 제거 → 빈 값 (원래 값은 Bronze에 남음)

df["available_at"] = df.observed_at + pd.Timedelta(hours=1)         # 공통 기준 B: 관측 1시간 뒤부터 사용

summary = pd.DataFrame(summary).groupby(["rule_id", "element", "status"], as_index=False).sum()
summary.to_csv(f"{OUT_DIR}/aws_qc_summary.csv", index=False, encoding="utf-8-sig")
print("\n[규칙별 건수]")
print(summary.to_string(index=False))

# ================================================================ 10. 저장: 월별 Silver 파일
cols = ["observed_at", "available_at", "station_id"] + NUM + [f"qc_{c}" for c in NUM] + [f"qc_{c}_flags" for c in NUM] + ["bronze_file"]
for mo, g in df.groupby("month"):
    g[cols].reset_index(drop=True).to_parquet(f"{OUT_DIR}/aws_silver_{mo}.parquet", index=False)
print(f"\n저장: {OUT_DIR}/aws_silver_YYYYMM.parquet {df.month.nunique()}개, {len(df):,}행")

# ================================================================ 11. 가용성 표 (관측소 × 월 × 요소)
hours_in_month = pd.Series({mo: pd.Period(mo[:4] + "-" + mo[4:], "M").days_in_month * 24 for mo in months})
av = []
for c in NUM:
    q = df[f"qc_{c}"]
    t = pd.DataFrame({"station_id": df.station_id, "month": df.month,
                      "normal": q == NORMAL, "missing": q == MISSING, "check": q == CHECK, "removed": q == REMOVE})
    a = t.groupby(["station_id", "month"]).agg(n_rows=("normal", "size"), n_normal=("normal", "sum"),
                                               n_missing=("missing", "sum"), n_check=("check", "sum"), n_removed=("removed", "sum")).reset_index()
    a.insert(2, "element", c)
    av.append(a)
av = pd.concat(av, ignore_index=True)
av["hours_in_month"] = av.month.map(hours_in_month)
av["value_share"] = ((av.n_normal + av.n_check) / av.hours_in_month).round(4)    # 그 달 시간 중 값이 남아 있는 비율
av.to_parquet(f"{OUT_DIR}/aws_availability.parquet", index=False)
av.to_csv(f"{OUT_DIR}/aws_availability.csv", index=False, encoding="utf-8-sig")
print(f"저장: {OUT_DIR}/aws_availability.parquet/.csv {len(av):,}행")
print("값이 하나도 없는 관측소×월 수:", av[av.value_share == 0].groupby("element").size().reindex(NUM, fill_value=0).to_dict())

# ================================================================ 12. 단위 표
units = pd.DataFrame([
    ("TA", "기온", "℃", "정시값"),
    ("WD", "풍향", "deg", "0~360. 풍속 0.2 m/s 미만이면 풍향은 의미 없음(WMO). Gold에서 제외"),
    ("WS", "풍속", "m/s", "정시 10분 평균"),
    ("RN_DAY", "일강수량", "mm", "00:01에 초기화되는 누적값. 00시 값은 전날 합계"),
    ("RN_HR1", "1시간 강수량", "mm", "지난 1시간 동안의 강수량"),
    ("HM", "상대습도", "%", "정시값"),
    ("PA", "현지기압", "hPa", "관측소 높이에서의 기압"),
    ("PS", "해면기압", "hPa", "현지기압·기온·높이로 해수면 높이로 환산한 값"),
], columns=["element", "name_ko", "unit", "note"])
units.to_csv(f"{OUT_DIR}/aws_units.csv", index=False, encoding="utf-8-sig")
print(f"저장: {OUT_DIR}/aws_units.csv")

# ================================================================ 13. 확인 출력
print("\n[요소별 최종 판정 (행)]")
fin = pd.DataFrame({c: df[f"qc_{c}"].value_counts().reindex([0, 1, 2, 3], fill_value=0) for c in NUM}).T
fin.columns = ["정상", "결측", "확인 필요", "제거"]
print(fin.to_string())
print("\n[산불에 중요한 극한값이 제거됐나]")
for c, cond, label in [("HM", V["HM"] < 15, "습도 15% 미만"), ("WS", V["WS"] >= 10, "풍속 10 m/s 이상"), ("TA", V["TA"] >= 35, "기온 35℃ 이상")]:
    q = df[f"qc_{c}"].to_numpy()[cond]
    print(f"  {label}: {int(cond.sum()):,}행 / 확인 필요 {int((q == CHECK).sum()):,} / 제거 {int((q == REMOVE).sum()):,}")