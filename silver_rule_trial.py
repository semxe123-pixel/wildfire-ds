"""4단계 규칙 시험: 확정한 QC 규칙을 실제 데이터에 적용해 보고 결과만 출력한다. 저장하지 않는다.

규칙의 두 원칙
- 실시간: 시각 t의 판단에는 t까지의 값(자기 관측소의 과거 값, 같은 시각까지의 이웃 값)만 쓴다
- 의심과 실패를 나눈다: 의심(2)은 값을 그대로 두고 표시만, 실패(3)는 값을 빈 값으로 바꿀 대상

출력
① 규칙별 건수 (2024·2025년 따로)      ② 요소별 최종 상태 (정상/결측/의심/실패)
③ 한 값이 여러 규칙에 걸린 경우        ④ 이웃 조건 민감도 (높이 차이 제한, 최소 이웃 수)
⑤ 기온 0.0 세부                       ⑥ 아직 정하지 않은 세부값 (습도 95% 경계, 이웃 바람 기준 등)
⑦ 산불에 중요한 극한값이 얼마나 걸리나  ⑧ 규칙별 실제 사례
"""
import sys
import warnings
import numpy as np
import pandas as pd
from silver_explore_common import load_aws, load_station_info, NUM

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.stdout.reconfigure(encoding="utf-8")         # 파일로 저장할 때 Windows 기본 인코딩(cp949)에 없는 글자(−) 때문에 멈추지 않게
pd.set_option("display.width", 220)
MAX_KM, K = 30, 3
LAPSE = 0.0065                                   # ℃/m, 100 m 높을 때마다 0.65℃ 낮음
SUSPECT, FAIL = 2, 3
rng = np.random.default_rng(0)

df = load_aws()
df["month"] = df.obs_time.dt.strftime("%Y%m")
df["year"] = df.obs_time.dt.year
info = load_station_info()
print(f"전체 {len(df):,}행, 관측소 {df.STN.nunique()}곳")
print(f"같은 관측소·시각이 두 번 나온 행: {int(df.duplicated(['STN', 'obs_time']).sum()):,}")

# ---------------------------------------------------------------- 준비
# 바로 앞 행이 같은 관측소의 정확히 1시간 전인가 (실시간 규칙은 과거만 보므로 이것만 필요)
cont = ((df.STN == df.STN.shift()) & (df.obs_time - df.obs_time.shift() == pd.Timedelta(hours=1))).to_numpy()


CLEAN = {}                                       # 물리한계에서 실패한 값을 뺀 값 (단계·이웃 비교용)


def clean(c):
    return CLEAN.get(c, df[c])


def change(c):
    """t의 값 − t−1의 값 (실패한 값은 빼고). 앞 시각이 없으면 빈 값"""
    v = clean(c)
    return (v - v.shift()).where(cont)


def run_so_far(c):
    """t까지 같은 값이 몇 시간째 이어지는가 (t 포함). 미래는 보지 않음"""
    v = df[c]
    same = cont & (v == v.shift()).to_numpy() & v.notna().to_numpy()
    rid = pd.Series((~same).cumsum(), index=df.index)
    return rid.groupby(rid).cumcount() + 1, rid


# 시각 × 관측소 표 (이웃 값을 빨리 찾기 위함)
times = pd.date_range("2024-01-01", "2025-12-31 23:00", freq="h", tz="Asia/Seoul")
stns = np.sort(df.STN.unique())
si = pd.Series(np.arange(len(stns)), index=stns)
r_t = pd.Series(np.arange(len(times)), index=times).reindex(df.obs_time).to_numpy()
r_s = si.reindex(df.STN).to_numpy()
MAT = {}


def matrix(c):
    if c not in MAT:
        M = np.full((len(times), len(stns)), np.nan, dtype="float32")
        M[r_t, r_s] = clean(c).to_numpy()
        MAT[c] = M
    return MAT[c]


# 관측소 × 월마다 30 km 안의 가까운 관측소 (가까운 순, 최대 10곳)와 높이 차
NB = {}
for mo, m in info.groupby("month"):
    m = m[m.STN.isin(stns)]
    lat, lon = np.radians(m.LAT.to_numpy()), np.radians(m.LON.to_numpy())
    km = 6371 * np.hypot((lon[:, None] - lon[None, :]) * np.cos(lat[:, None]), lat[:, None] - lat[None, :])
    ids, ht = m.STN.to_numpy(), m.HT.to_numpy()
    for i, s in enumerate(ids):
        order = [j for j in np.argsort(km[i])[1:11] if km[i][j] <= MAX_KM]
        NB[(s, mo)] = (si[ids[order]].to_numpy(), ht[order] - ht[i])


def nb_stat(c, rows, mode="level", lapse=0.0, max_dh=None, min_n=1):
    """rows(행 번호)마다 이웃 최대 K곳의 중앙값과 이웃 수.
    mode="level": 같은 시각 t의 값 (lapse로 높이 보정)
    mode="change": 같은 시각의 1시간 변화량 (t − t−1)
    max_dh: 높이 차이가 이보다 큰 이웃은 뺌 / min_n: 이웃이 이보다 적으면 빈 값"""
    rows = np.asarray(rows)
    med = np.full(len(rows), np.nan, dtype="float32")
    cnt = np.zeros(len(rows), dtype="int8")
    if len(rows) == 0:
        return med, cnt
    M = matrix(c)
    sub = df.iloc[rows]
    for (s, mo), g in sub.groupby(["STN", "month"]):
        if (s, mo) not in NB:
            continue
        cols, dh = NB[(s, mo)]
        if max_dh is not None:
            ok_h = np.abs(dh) <= max_dh
            cols, dh = cols[ok_h], dh[ok_h]
        if len(cols) == 0:
            continue
        t = r_t[g.index]
        if mode == "level":
            vals = M[t][:, cols] + lapse * dh
        else:
            prev = np.where(t > 0, t - 1, 0)
            vals = M[t][:, cols] - M[prev][:, cols]
            vals[t == 0] = np.nan
        ok = ~np.isnan(vals)
        keep = ok & (np.cumsum(ok, axis=1) <= K)
        vals = np.where(keep, vals, np.nan)
        pos = np.searchsorted(rows, g.index)
        med[pos] = np.nanmedian(vals, axis=1)
        cnt[pos] = keep.sum(axis=1)
    med[cnt < min_n] = np.nan
    return med, cnt


def rows_of(mask):
    return np.flatnonzero(np.asarray(mask, dtype=bool))


def to_mask(rows):
    m = np.zeros(len(df), dtype=bool)
    m[rows] = True
    return m


# 규칙 결과를 모으는 곳: (규칙 이름, 걸린 요소들, 상태, 행 마스크)
RULES = []


def add(name, elems, status, mask):
    RULES.append((name, elems, status, np.asarray(mask, dtype=bool)))


# ---------------------------------------------------------------- 코드 2: 물리한계
add("TA_범위밖", ["TA"], FAIL, (df.TA < -40) | (df.TA > 60))                 # 센서 측정 범위 −40~60
add("TA_45초과", ["TA"], SUSPECT, (df.TA > 45) & (df.TA <= 60))
add("HM_범위밖", ["HM"], FAIL, (df.HM < 0) | (df.HM > 100))
add("HM_0", ["HM"], SUSPECT, df.HM == 0)
add("RN_HR1_음수", ["RN_HR1"], FAIL, df.RN_HR1 < 0)
add("RN_DAY_음수", ["RN_DAY"], FAIL, df.RN_DAY < 0)
add("WS_음수", ["WS"], FAIL, df.WS < 0)
add("WS_60초과", ["WS"], SUSPECT, df.WS > 60)
add("WD_범위밖", ["WD"], FAIL, (df.WD < 0) | (df.WD > 360))
add("PS_범위밖", ["PS"], SUSPECT, (df.PS < 900) | (df.PS > 1080))

ht = df[["STN", "month"]].merge(info[["STN", "month", "HT"]], on=["STN", "month"], how="left").HT.to_numpy()
pa_expected = 1013.25 * (1 - 2.25577e-5 * ht) ** 5.25588                    # 표준대기에서 높이별 기압
add("PA_높이차100", ["PA"], FAIL, np.abs(df.PA.to_numpy() - pa_expected) > 100)

# 물리한계에서 실패한 값은 단계·이웃 비교에서 뺌
# (예: 1007.6 → 500 → 1010.5에서 500을 빼지 않으면, 정상인 1010.5가 "500에서 뛰었다"로 걸림)
for c in NUM:
    bad = np.zeros(len(df), dtype=bool)
    for name, elems, st, m in RULES:
        if c in elems and st == FAIL:
            bad |= m
    CLEAN[c] = df[c].where(~bad)

# ---------------------------------------------------------------- 코드 3: 단계 (실시간)
# 앞 시각보다 기준 넘게 변했는데, 같은 시각 이웃의 변화 중앙값이 같은 방향으로 기준의 절반을 넘지 않음
STEP = {"TA": 5, "HM": 50, "PA": 5, "PS": 5}
STEP_NO_NB = {}
for c, thr in STEP.items():
    d = change(c).to_numpy()
    cand = rows_of(np.abs(d) > thr)
    nbd, n = nb_stat(c, cand, mode="change")
    with_nb = n > 0
    weather = (np.sign(nbd) == np.sign(d[cand])) & (np.abs(nbd) > thr / 2)
    add(f"{c}_단계", [c], SUSPECT, to_mask(cand[with_nb & ~weather]))
    STEP_NO_NB[c] = (len(cand), int((~with_nb).sum()))

# ---------------------------------------------------------------- 코드 4: 지속성 (실시간, "지금까지 몇 시간째")
# 기온 0.0: 1~2시간째는 정상, 3~5시간째 + 이웃과 5℃ 넘게 차이 = 의심, 6시간째 이상 + 차이 = 실패
ta_run, _ = run_so_far("TA")
z_rows = rows_of((df.TA == 0).to_numpy() & (ta_run.to_numpy() >= 3))
z_nb, z_n = nb_stat("TA", z_rows, mode="level", lapse=LAPSE)
z_len = ta_run.to_numpy()[z_rows]
far = np.abs(z_nb) > 5
add("TA_0.0_3~5시간", ["TA"], SUSPECT, to_mask(z_rows[far & (z_len <= 5)]))
add("TA_0.0_6시간+", ["TA"], FAIL, to_mask(z_rows[far & (z_len >= 6)]))
add("TA_0.0_이웃없음_12시간+", ["TA"], SUSPECT, to_mask(z_rows[(z_n == 0) & (z_len >= 12)]))

# 습도: 95% 미만인 같은 값이 24시간째 이상
hm_run, _ = run_so_far("HM")
add("HM_고착24", ["HM"], SUSPECT, (df.HM < 95).to_numpy() & (hm_run.to_numpy() >= 24))

# 풍속 0: 24시간째 이상 + 지난 24시간 중 75% 이상에서 이웃 풍속 중앙값 ≥ 2 m/s
ws_run, ws_rid = run_so_far("WS")
calm = (df.WS == 0).to_numpy()
total = ws_run.groupby(ws_rid).transform("max").to_numpy()                 # 후보를 고르는 데만 씀 (판단에는 안 씀)
c_rows = rows_of(calm & (total >= 24))
ws_nb, ws_n = nb_stat("WS", c_rows, mode="level")
WS_GRID = {}
for nb_thr in (1, 2):
    for share in (0.5, 0.75):
        windy = pd.Series((ws_nb >= nb_thr).astype(float))
        frac = windy.groupby(ws_rid.to_numpy()[c_rows]).transform(lambda x: x.rolling(24, min_periods=24).mean()).to_numpy()
        hit = (ws_run.to_numpy()[c_rows] >= 24) & (frac >= share)
        WS_GRID[(nb_thr, share)] = c_rows[hit]
add("WS_0고착24_이웃바람", ["WS"], SUSPECT, to_mask(WS_GRID[(2, 0.75)]))

# 기압: 같은 값이 12시간째 이상 (하루 두 번 약 1 hPa씩 오르내리므로 물리적으로 어려움)
for c in ("PA", "PS"):
    r, _ = run_so_far(c)
    add(f"{c}_고착12", [c], FAIL, df[c].notna().to_numpy() & (r.to_numpy() >= 12))

# ---------------------------------------------------------------- 코드 5: 내적 일치성
add("RN_HR1>RN_DAY", ["RN_HR1", "RN_DAY"], SUSPECT, df.RN_HR1 > df.RN_DAY + 0.05)
dec = cont & (df.RN_DAY < df.RN_DAY.shift() - 0.05).to_numpy()
add("RN_DAY_감소(01시외)", ["RN_DAY"], SUSPECT, dec & (df.obs_time.dt.hour != 1).to_numpy())

# ================================================================ ① 규칙별 건수
print("\n=== ① 규칙별 건수 (값이 있는 행만. 연도별은 10만 행당) ===")
n_year = df.groupby("year").size()
rows = []
for name, elems, st, m in RULES:
    m = m & df[elems[0]].notna().to_numpy()
    per = df[m].groupby("year").size().reindex(n_year.index, fill_value=0) / n_year * 1e5
    rows.append({"규칙": name, "요소": ",".join(elems), "상태": "실패" if st == FAIL else "의심",
                 "행": int(m.sum()), "관측소": df.STN[m].nunique(),
                 **{f"{y}(10만당)": round(v, 2) for y, v in per.items()}})
print(pd.DataFrame(rows).to_string(index=False))
print("단계 규칙 후보 중 이웃이 없어 판단하지 못한 행:", {c: f"{a - b:,}/{a:,} 판단, {b:,} 이웃없음" for c, (a, b) in STEP_NO_NB.items()})

# ================================================================ ② 요소별 최종 상태
print("\n=== ② 요소별 최종 상태 (여러 규칙에 걸리면 실패 > 의심) ===")
STATUS = {}
for c in NUM:
    st = np.where(df[c].isna(), 1, 0)
    for name, elems, s, m in RULES:
        if c in elems:
            hit = m & df[c].notna().to_numpy()
            st = np.where(hit, np.maximum(st, s), st)
    STATUS[c] = st
cnt = pd.DataFrame({c: pd.Series(STATUS[c]).value_counts().reindex([0, 1, 2, 3], fill_value=0) for c in NUM}).T
cnt.columns = ["정상", "결측", "의심", "실패"]
cnt["의심%"] = (cnt["의심"] / len(df) * 100).round(3)
cnt["실패%"] = (cnt["실패"] / len(df) * 100).round(3)
print(cnt.to_string())

# ================================================================ ③ 겹침
print("\n=== ③ 한 값이 두 규칙 이상에 걸린 경우 ===")
for c in NUM:
    names = [(n, m & df[c].notna().to_numpy()) for n, e, s, m in RULES if c in e]
    if len(names) < 2:
        continue
    B = np.column_stack([m for _, m in names])
    multi = B.sum(axis=1) >= 2
    if multi.any():
        combo = pd.Series(["+".join(n for (n, _), b in zip(names, row) if b) for row in B[multi]]).value_counts().head(5)
        print(f"[{c}] {int(multi.sum()):,}행:", combo.to_dict())
    else:
        print(f"[{c}] 없음")

# ================================================================ ④ 민감도
print("\n=== ④ 이웃 조건 민감도: 높이 차이 제한 × 최소 이웃 수 ===")
d_ta = change("TA").to_numpy()
cand = rows_of(np.abs(d_ta) > 5)
res = []
for max_dh in (None, 500, 300, 100):
    for min_n in (1, 3):
        nbd, n = nb_stat("TA", cand, "change", max_dh=max_dh, min_n=min_n)
        ok = ~np.isnan(nbd)
        weather = (np.sign(nbd) == np.sign(d_ta[cand])) & (np.abs(nbd) > 2.5)
        zn, _ = nb_stat("TA", z_rows, "level", LAPSE, max_dh=max_dh, min_n=min_n)
        zf = np.abs(zn) > 5
        res.append({"높이차 제한(m)": max_dh or "없음", "최소 이웃": min_n,
                    "TA단계_판단가능": int(ok.sum()), "TA단계_의심": int((ok & ~weather).sum()),
                    "TA0.0_판단가능": int((~np.isnan(zn)).sum()),
                    "TA0.0_의심(3~5)": int((zf & (z_len <= 5)).sum()), "TA0.0_실패(6+)": int((zf & (z_len >= 6)).sum())})
print(pd.DataFrame(res).to_string(index=False))

# ================================================================ ⑤ 기온 0.0 세부
print("\n=== ⑤ 기온 0.0: '지금까지 몇 시간째'별 이웃과 5℃ 넘게 차이 나는 행 ===")
all_z = rows_of((df.TA == 0).to_numpy())
azn, _ = nb_stat("TA", all_z, "level", LAPSE)
g = pd.cut(ta_run.to_numpy()[all_z], [0, 2, 5, 11, 47, 10**6], labels=["1~2", "3~5", "6~11", "12~47", "48+"])
t = pd.DataFrame({"g": g, "far": np.abs(azn) > 5, "has": ~np.isnan(azn),
                  "winter": df.obs_time.dt.month.to_numpy()[all_z] % 12 < 3})
print(t.groupby(["g", "winter"], observed=True).agg(행=("far", "size"), 이웃있음=("has", "sum"), 차이5초과=("far", "sum")).to_string())

# ================================================================ ⑥ 아직 정하지 않은 세부값
print("\n=== ⑥ 세부값 ===")
hv, hr = df.HM.to_numpy(), hm_run.to_numpy()
print("[습도 고착 24시간째 이상] 값 구간별 행 수:",
      {"95 미만": int(((hv < 95) & (hr >= 24)).sum()), "95~99.9": int(((hv >= 95) & (hv < 100) & (hr >= 24)).sum()),
       "100": int(((hv == 100) & (hr >= 24)).sum())})
top = pd.Series(hv[(hv >= 95) & (hv < 100) & (hr >= 24)]).value_counts().head(5)
print("   95~99.9 구간에서 많이 고정된 값:", top.to_dict())

print("[풍속 0 고착] 후보 행(24시간 이상 구간) {:,}, 이웃 없음 {:,}".format(len(c_rows), int((ws_n == 0).sum())))
for (thr, share), r in WS_GRID.items():
    print(f"   이웃 중앙값 ≥ {thr} m/s 인 시간이 {int(share * 100)}% 이상: {len(r):,}행, 관측소 {df.STN.iloc[r].nunique()}곳")

d02 = dec & (df.obs_time.dt.hour == 2).to_numpy()
was01 = pd.Series(dec).shift(fill_value=False).to_numpy()                   # 바로 앞 행(01시)도 감소였나
print(f"[일강수 02시 감소] {int(d02.sum())}행 / 그중 01시에 감소가 없었던 행(초기화가 늦게 반영) {int((d02 & ~was01).sum())}행")
print("   관측소별 상위 10:", df.STN[d02].value_counts().head(10).to_dict())

j = rows_of((df.STN == 352).to_numpy() & (df.HM < 5).to_numpy())
jn, _ = nb_stat("HM", j, "level")
print(f"[352 장흥면 습도 5% 미만] {len(j)}행 / 같은 시각 이웃 습도 중앙값 분포:",
      pd.Series(jn).describe()[["count", "min", "50%", "max"]].round(1).to_dict())
print("   월별:", df.month.iloc[j].value_counts().sort_index().to_dict())

ta, pa, ps = df.TA.to_numpy(), df.PA.to_numpy(), df.PS.to_numpy()
ps_est = pa * (1 - (LAPSE * ht) / (ta + LAPSE * ht + 273.15)) ** -5.257    # 현지기압·기온·높이로 계산한 해면기압
dd = pd.Series(ps - ps_est).dropna()
print(f"[현지기압→해면기압 계산값과 실제 PS 차이] {len(dd):,}행 / 분포:", dd.quantile([0.001, 0.01, 0.5, 0.99, 0.999]).round(2).to_dict())
print("   |차이| > 3, 5, 10 hPa:", {t: int((dd.abs() > t).sum()) for t in (3, 5, 10)})

p_rows = rows_of(df.PS.notna().to_numpy())
pn, _ = nb_stat("PS", p_rows, "level")
pdiff = pd.Series(ps[p_rows] - pn).dropna()
print(f"[해면기압 − 이웃 중앙값] {len(pdiff):,}행 / 분포:", pdiff.quantile([0.001, 0.01, 0.5, 0.99, 0.999]).round(2).to_dict())
print("   |차이| > 3, 5, 10 hPa:", {t: int((pdiff.abs() > t).sum()) for t in (3, 5, 10)})

# ================================================================ ⑦ 산불에 중요한 극한값
print("\n=== ⑦ 산불에 중요한 극한값이 규칙에 얼마나 걸리나 ===")
for c, cond, label in [("HM", df.HM < 15, "습도 15% 미만"), ("WS", df.WS >= 10, "풍속 10 m/s 이상"), ("TA", df.TA >= 35, "기온 35℃ 이상")]:
    m = cond.to_numpy()
    st = STATUS[c][m]
    print(f"[{label}] {int(m.sum()):,}행 / 정상 {int((st == 0).sum()):,} / 의심 {int((st == 2).sum()):,} / 실패 {int((st == 3).sum()):,}")

# ================================================================ ⑧ 사례
print("\n=== ⑧ 규칙별 무작위 사례 (앞뒤 2시간 값은 확인용. 판단에는 t까지만 씀) ===")
for name, elems, st, m in RULES:
    c = elems[0]
    r = rows_of(m & df[c].notna().to_numpy())
    if len(r) == 0:
        continue
    pick = np.sort(rng.choice(r, min(3, len(r)), replace=False))
    nbv, _ = nb_stat(c, pick, "level", LAPSE if c == "TA" else 0.0)
    print(f"[{name}]")
    for i, nv in zip(pick, nbv):
        w = df.loc[max(i - 2, 0):i + 2]
        w = w[w.STN == df.at[i, "STN"]]
        vals = " → ".join(("[" if k == i else "") + f"{v:g}" + ("]" if k == i else "") for k, v in zip(w.index, w[c]))
        print(f"   STN {df.at[i, 'STN']} {df.at[i, 'obs_time']:%Y-%m-%d %H시}  {c}: {vals}  / 같은 시각 이웃 중앙값 {nv:.1f}")