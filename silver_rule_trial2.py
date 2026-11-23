"""4단계 규칙 시험 2: 시험 1에서 남은 7가지만 확인한다. 저장하지 않고 출력만 한다.

① 기온 단계: "자기 변화 − 이웃 변화"의 차이로 판단하면 몇 건이 걸리나 (기준 5·6·8·10℃)
② 단계 규칙의 '돌아온 값' 문제: 튄 값 다음에 원래 수준으로 돌아온 정상 값이 걸리는 경우를 빼면 몇 건이 남나
③ 습도 95% 이상 고착: 같은 시각 이웃은 건조한가
④ 아주 낮은 습도: 높이가 비슷한 이웃보다 50%p 넘게 낮은가 (352 장흥면 등)
⑤ 일강수 01시 초기화 지연: 01시 일강수 > 01시 1시간 강수
⑥ 현지기압·해면기압 불일치, 해면기압 이웃 차이: 어느 관측소·날짜에 몰리나
⑦ 풍속 0 고착: 느슨한 기준(이웃 1 m/s, 50%)에서만 걸리는 사례
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
add("HM_0", ["HM"], FAIL, df.HM == 0)                                 # 시험 1: 18건 모두 00시, 이웃 77~85% → 실패
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


def show(c, rows, n=3, label="", lapse=0.0, max_dh=None):
    """무작위 사례: 앞뒤 2시간 값(확인용)과 같은 시각 이웃 중앙값"""
    rows = np.asarray(rows)
    if len(rows) == 0:
        print(f"   {label} 사례 없음")
        return
    pick = np.sort(rng.choice(rows, min(n, len(rows)), replace=False))
    nbv, _ = nb_stat(c, pick, "level", lapse, max_dh=max_dh)
    for i, nv in zip(pick, nbv):
        w = df.loc[max(i - 2, 0):i + 2]
        w = w[w.STN == df.at[i, "STN"]]
        vals = " → ".join(("[" if k == i else "") + f"{v:g}" + ("]" if k == i else "") for k, v in zip(w.index, w[c]))
        print(f"   {label}STN {df.at[i, 'STN']} {df.at[i, 'obs_time']:%Y-%m-%d %H시}  {c}: {vals}  / 이웃 {nv:.1f}")


# ================================================================ ① 기온 단계: 변화량 차이
print("\n=== ① 기온 단계: |자기 1시간 변화 − 이웃 1시간 변화 중앙값| ===")
d_ta = change("TA").to_numpy()
smp = np.sort(rng.choice(rows_of(~np.isnan(d_ta)), 500_000, replace=False)) if np.isfinite(d_ta).sum() > 500_000 else rows_of(~np.isnan(d_ta))
nbd_s, _ = nb_stat("TA", smp, "change")
gap = pd.Series(np.abs(d_ta[smp] - nbd_s)).dropna()
print(f"무작위 {len(gap):,}행의 차이 분포:", gap.quantile([0.5, 0.9, 0.99, 0.999, 0.9999]).round(2).to_dict())
cand = rows_of(np.abs(d_ta) > 5)
nbd, n = nb_stat("TA", cand, "change")
g_c = np.abs(d_ta[cand] - nbd)
old = (np.sign(nbd) == np.sign(d_ta[cand])) & (np.abs(nbd) > 2.5)
print(f"자기 변화 > 5℃ 후보 {len(cand):,}행 / 지금 규칙(이웃이 같은 방향 2.5℃ 미만)으로 의심: {int(((n > 0) & ~old).sum()):,}")
for T in (5, 6, 8, 10):
    print(f"   차이 > {T:>2}℃ 로 바꾸면 의심: {int((g_c > T).sum()):,}")
print(" 지금 규칙에는 걸리지만 '차이 > 8℃'에는 안 걸리는 사례 (진짜 날씨인지 확인):")
show("TA", cand[(n > 0) & ~old & ~(g_c > 8)], 5, lapse=LAPSE)
print(" '차이 > 8℃'에 걸리는 사례:")
show("TA", cand[g_c > 8], 5, lapse=LAPSE)

# ================================================================ ② 돌아온 값 빼기
print("\n=== ② 튄 값 다음에 원래 수준으로 돌아온 값 (t−1이 걸렸고, t가 t−2와 기준 이내) ===")
STEP = {"TA": 8, "HM": 50, "PA": 5, "PS": 5}
for c, thr in STEP.items():
    v = clean(c).to_numpy()
    d = change(c).to_numpy()
    cand = rows_of(np.abs(d) > thr)
    nbd, n = nb_stat(c, cand, "change")
    if c == "TA":
        hit = cand[np.abs(d[cand] - nbd) > thr]
    else:
        hit = cand[(n > 0) & ~((np.sign(nbd) == np.sign(d[cand])) & (np.abs(nbd) > thr / 2))]
    F = to_mask(hit)
    cont2 = cont & np.roll(cont, 1)                                           # t−1, t−2가 모두 정확히 1시간 간격
    back = cont2 & (np.abs(v - np.roll(v, 2)) <= thr)
    ret = F & np.roll(F, 1) & back
    print(f"[{c} 기준 {thr}] 걸림 {int(F.sum()):,} / 그중 돌아온 값 {int(ret.sum()):,} → 남는 의심 {int((F & ~ret).sum()):,}")
    show(c, rows_of(ret), 2, "돌아온 값 ")

# ================================================================ ③ 습도 95% 이상 고착
print("\n=== ③ 습도 95~99.9% 같은 값 24시간째 이상: 같은 시각 이웃 습도 ===")
hm_run, _ = run_so_far("HM")
hv = df.HM.to_numpy()
sat = rows_of((hv >= 95) & (hv < 100) & (hm_run.to_numpy() >= 24))
sn, sc = nb_stat("HM", sat, "level")
print(f"{len(sat):,}행 / 이웃 없음 {int((sc == 0).sum()):,} / 이웃 중앙값 분포:", pd.Series(sn).describe()[["min", "25%", "50%", "75%"]].round(1).to_dict())
for T in (90, 80, 70):
    m = sn < T
    print(f"   이웃 중앙값 < {T}%: {int(m.sum()):,}행, 관측소별 {df.STN.iloc[sat[m]].value_counts().head(5).to_dict()}")
show("HM", sat[sn < 70], 3, "이웃<70 ")

# ================================================================ ④ 아주 낮은 습도
print("\n=== ④ 습도 10% 미만: 이웃보다 50%p 넘게 낮은가 ===")
low = rows_of(hv < 10)
for max_dh in (None, 300):
    ln, lc = nb_stat("HM", low, "level", max_dh=max_dh)
    m = (ln - hv[low]) > 50
    print(f"[높이차 제한 {max_dh or '없음'}] 10% 미만 {len(low):,}행 / 이웃 있음 {int((lc > 0).sum()):,} / 50%p 넘게 낮음 {int(m.sum()):,}행")
    print("   관측소별:", df.STN.iloc[low[m]].value_counts().head(8).to_dict())
show("HM", low[m], 4, "높이 비슷한 이웃보다 50%p↓ ", max_dh=300)

# ================================================================ ⑤ 일강수 01시 초기화 지연
print("\n=== ⑤ 01시 일강수 > 01시 1시간 강수 + 0.5 (01시 일강수는 00:01부터 쌓인 양이라 1시간 강수와 거의 같아야 함) ===")
h1 = (df.obs_time.dt.hour == 1).to_numpy()
late = h1 & (df.RN_DAY > df.RN_HR1 + 0.5).to_numpy()
print(f"{int(late.sum()):,}행, 관측소 {df.STN[late].nunique()}곳 / 관측소별:", df.STN[late].value_counts().head(10).to_dict())
nxt_dec = np.roll((df.RN_DAY < df.RN_DAY.shift() - 0.05).to_numpy() & cont, -1)
print(f"   그중 바로 다음 02시에 일강수가 줄어든 행: {int((late & nxt_dec).sum()):,}")
show("RN_DAY", rows_of(late), 4)

# ================================================================ ⑥ 기압 불일치
print("\n=== ⑥ 기압 ===")
ta, pa, ps = df.TA.to_numpy(), clean("PA").to_numpy(), df.PS.to_numpy()
ps_est = pa * (1 - (LAPSE * ht) / (ta + LAPSE * ht + 273.15)) ** -5.257
mis = np.abs(ps - ps_est) > 10
print(f"[현지→해면 계산값과 PS 차이 > 10 hPa] {int(mis.sum()):,}행 / 관측소별:", df.STN[mis].value_counts().head(8).to_dict())
print("   날짜별 상위 5:", df.obs_time[mis].dt.strftime("%Y-%m-%d").value_counts().head(5).to_dict())
show("PS", rows_of(mis), 3)
p_rows = rows_of(df.PS.notna().to_numpy())
pn, _ = nb_stat("PS", p_rows, "level")
far = p_rows[np.abs(ps[p_rows] - pn) > 10]
print(f"[해면기압 − 이웃 중앙값 > 10 hPa] {len(far):,}행 / 관측소별:", df.STN.iloc[far].value_counts().head(8).to_dict())
print("   날짜별 상위 5 (태풍 날짜에 몰리면 진짜 기압차일 수 있음):", df.obs_time.iloc[far].dt.strftime("%Y-%m-%d").value_counts().head(5).to_dict())
show("PS", far, 3)

# ================================================================ ⑦ 풍속 0 고착
print("\n=== ⑦ 풍속 0 고착: 느슨한 기준에서만 걸리는 사례 ===")
ws_run, ws_rid = run_so_far("WS")
total = ws_run.groupby(ws_rid).transform("max").to_numpy()
c_rows = rows_of((df.WS == 0).to_numpy() & (total >= 24))
ws_nb, _ = nb_stat("WS", c_rows, "level")
res = {}
for thr, share in ((1, 0.5), (2, 0.75)):
    windy = pd.Series((ws_nb >= thr).astype(float))
    frac = windy.groupby(ws_rid.to_numpy()[c_rows]).transform(lambda x: x.rolling(24, min_periods=24).mean()).to_numpy()
    res[(thr, share)] = set(c_rows[(ws_run.to_numpy()[c_rows] >= 24) & (frac >= share)])
only = np.array(sorted(res[(1, 0.5)] - res[(2, 0.75)]))
print(f"이웃 1 m/s·50%에서만 걸림 {len(only):,}행 / 관측소별:", df.STN.iloc[only].value_counts().to_dict() if len(only) else {})
show("WS", only, 5)