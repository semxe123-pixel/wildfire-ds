"""3-4 기상 탐색: 이웃 비교 기준 + 내적 일치성 + 가용성. 저장하지 않고 출력만 한다.

① 이웃과의 차이가 "정상일 때" 얼마나 되나 (기온·습도). 이웃 = 30km 안의 가까운 관측소 최대 3곳
   기온은 높이 차이를 보정함: 100m 높을 때마다 0.65℃ 낮아지는 것으로 봄
② 기온 0.0인 행: 이웃 기온과 얼마나 다른가 (반복 길이별, 계절별)
③ 습도: 100%가 아닌 값의 반복 길이, 시각(시)별 "이웃과 크게 다른 값"의 분포 (00시 오류 확인)
④ 내적 일치성: 1시간 강수 > 일 강수, 일 강수가 하루 중에 줄어듦, 풍속 0인데 풍향 있음
⑤ 가용성: 관측소 × 월 × 요소별로 값이 있는 비율
"""
import warnings
import numpy as np
import pandas as pd
from silver_explore_common import load_aws, load_station_info, q, NUM

warnings.filterwarnings("ignore", category=RuntimeWarning)   # 이웃이 모두 빈 값인 행의 중앙값 경고
pd.set_option("display.width", 220)
MAX_KM, K = 30, 3
LAPSE = 0.0065                       # ℃/m

df = load_aws()
df["month"] = df.obs_time.dt.strftime("%Y%m")
info = load_station_info()

# 시각 × 관측소 표 (빠른 조회용)
times = pd.date_range("2024-01-01", "2025-12-31 23:00", freq="h", tz="Asia/Seoul")
stns = np.sort(df.STN.unique())
ti = pd.Series(np.arange(len(times)), index=times)
si = pd.Series(np.arange(len(stns)), index=stns)
r_t = ti.reindex(df.obs_time).to_numpy()
r_s = si.reindex(df.STN).to_numpy()


def matrix(c):
    M = np.full((len(times), len(stns)), np.nan, dtype="float32")
    M[r_t, r_s] = df[c].to_numpy()
    return M


# 관측소 × 월마다 30km 안의 가까운 관측소 (가까운 순, 최대 10곳)
NB = {}
for mo, m in info.groupby("month"):
    m = m[m.STN.isin(stns)]
    lat, lon = np.radians(m.LAT.to_numpy()), np.radians(m.LON.to_numpy())
    km = 6371 * np.hypot((lon[:, None] - lon[None, :]) * np.cos(lat[:, None]), lat[:, None] - lat[None, :])
    ids, ht = m.STN.to_numpy(), m.HT.to_numpy()
    for i, s in enumerate(ids):
        order = [j for j in np.argsort(km[i])[1:11] if km[i][j] <= MAX_KM]
        NB[(s, mo)] = (si[ids[order]].to_numpy(), ht[order] - ht[i])     # 이웃 열 번호, 높이 차(이웃 − 나)


def neighbor_median(M, rows, lapse=0.0):
    """rows(df의 행 번호)마다 이웃 최대 K곳의 같은 시각 값의 중앙값. 높이 보정 선택"""
    out = np.full(len(rows), np.nan, dtype="float32")
    sub = df.iloc[rows]
    for (s, mo), g in sub.groupby(["STN", "month"]):
        if (s, mo) not in NB or len(NB[(s, mo)][0]) == 0:
            continue
        cols, dh = NB[(s, mo)]
        vals = M[r_t[g.index]][:, cols] + lapse * dh          # 이웃 값을 내 높이로 보정
        ok = ~np.isnan(vals)
        keep = ok & (np.cumsum(ok, axis=1) <= K)               # 값이 있는 이웃 중 가까운 K곳
        vals = np.where(keep, vals, np.nan)
        pos = np.searchsorted(rows, g.index)
        with np.errstate(all="ignore"):
            out[pos] = np.nanmedian(vals, axis=1)
    return out


rng = np.random.default_rng(0)
TA = matrix("TA")
HM = matrix("HM")

# ① 정상일 때의 이웃 차이 (무작위 20만 행)
print("=== ① 이웃과의 차이: 무작위 20만 행 (정상 수준을 보기 위함) ===")
sample = np.sort(rng.choice(len(df), min(len(df), 200_000), replace=False))
for c, M, lapse in [("TA", TA, LAPSE), ("HM", HM, 0.0)]:
    nm = neighbor_median(M, sample, lapse)
    d = np.abs(df[c].to_numpy()[sample] - nm)
    d = d[~np.isnan(d)]
    print(f"[{c}] 이웃 비교 가능 {len(d):,}행 / |차이| 분포:", q(pd.Series(d), ps=(0.5, 0.9, 0.99, 0.999, 0.9999, 1)))
    for t in ((2, 3, 5, 8) if c == "TA" else (10, 20, 30, 50)):
        print(f"   |차이| > {t}: {np.mean(d > t) * 100:.3f}%")

# ② 기온 0.0
print("\n=== ② 기온 0.0인 행 vs 이웃 기온 (높이 보정) ===")
cont = (df.STN == df.STN.shift()) & (df.obs_time - df.obs_time.shift() == pd.Timedelta(hours=1))
z = (df.TA == 0)
run_id = (~(cont & z & z.shift(fill_value=False))).cumsum()
run_len = z.groupby(run_id).transform("sum")
zr = np.flatnonzero(z.to_numpy())
nm0 = neighbor_median(TA, zr, LAPSE)
t0 = pd.DataFrame({"len": run_len.to_numpy()[zr], "nb": nm0, "mon": df.obs_time.dt.month.to_numpy()[zr]})
t0["season"] = np.where(t0.mon.isin([12, 1, 2]), "겨울(12~2월)", "그 외")
t0["len_g"] = pd.cut(t0.len, [0, 2, 5, 11, 10**6], labels=["1~2", "3~5", "6~11", "12+"])
print(f"기온 0.0 {len(t0):,}행 / 이웃 비교 가능 {int(t0.nb.notna().sum()):,}행")
tab = t0.dropna(subset=["nb"]).assign(far=lambda x: x.nb.abs()).groupby(["season", "len_g"], observed=True).far
print(pd.DataFrame({"행": tab.size(), "이웃 |기온| 중앙값": tab.median().round(1),
                    "> 3℃": tab.apply(lambda x: int((x > 3).sum())), "> 5℃": tab.apply(lambda x: int((x > 5).sum())),
                    "> 8℃": tab.apply(lambda x: int((x > 8).sum()))}).to_string())
print(f"이웃 없음(30km 안 관측소 없음 또는 값 없음): {int(t0.nb.isna().sum()):,}행, 그중 반복 3시간 이상 {int((t0.nb.isna() & (t0.len >= 3)).sum()):,}행")

# ③ 습도
print("\n=== ③ 습도 ===")
v = df.HM
same = cont & (v == v.shift()) & v.notna() & (v < 99)
rid = (~same).cumsum()
g = pd.DataFrame({"rid": rid, "STN": df.STN, "t": df.obs_time, "v": v})[v.notna() & (v < 99)]
r = g.groupby("rid").agg(STN=("STN", "first"), start=("t", "first"), length=("t", "size"), value=("v", "first"))
print("99% 미만 값의 반복 길이:", pd.cut(r.length, [0, 2, 5, 11, 23, 47, 10**6], labels=["1~2", "3~5", "6~11", "12~23", "24~47", "48+"]).value_counts().sort_index().to_dict())
print(r.sort_values("length", ascending=False).head(8).to_string(index=False))
hr = np.flatnonzero(df.HM.notna().to_numpy())
hr = np.sort(rng.choice(hr, min(len(hr), 2_000_000), replace=False))
nmh = neighbor_median(HM, hr)
dh = df.HM.to_numpy()[hr] - nmh
hh = df.obs_time.dt.hour.to_numpy()[hr]
ok = ~np.isnan(dh)
print("\n시각(시)별 '이웃보다 50%p 이상 낮은 습도' 비율 (무작위 200만 행 중):")
s = pd.Series(dh[ok] < -50).groupby(hh[ok]).mean().mul(100).round(3)
print("   ", s.to_dict())

# ④ 내적 일치성
print("\n=== ④ 내적 일치성 ===")
m1 = df.RN_HR1 > df.RN_DAY + 0.05
print(f"1시간 강수 > 일 강수: {int(m1.sum()):,}행 / 시각별:", df[m1].obs_time.dt.hour.value_counts().sort_index().to_dict())
dec = cont & (df.RN_DAY < df.RN_DAY.shift() - 0.05)
print(f"일 강수가 1시간 전보다 줄어듦: {int(dec.sum()):,}행 / 시각별 상위 5:", df[dec].obs_time.dt.hour.value_counts().head(5).to_dict())
print(f"풍속 0인데 풍향 > 0: {int(((df.WS == 0) & (df.WD > 0)).sum()):,}행 / 풍속 > 0인데 풍향 0: {int(((df.WS > 0) & (df.WD == 0)).sum()):,}행")

# ⑤ 가용성
print("\n=== ⑤ 가용성: 관측소 × 월 × 요소 ===")
hours = df.groupby("month").obs_time.nunique()
av = df.groupby(["STN", "month"])[NUM].count().div(hours, axis=0, level="month")
print(f"관측소×월 {len(av):,}개")
print("요소별 '값이 있는 비율' 분포 (관측소×월 기준):")
print(av.describe(percentiles=[0.01, 0.05, 0.5]).T[["min", "1%", "5%", "50%"]].round(3).to_string())
print("값이 있는 비율 < 50%인 관측소×월 수:", (av < 0.5).sum().to_dict())
print("값이 하나도 없는 관측소×월 수:", (av == 0).sum().to_dict())