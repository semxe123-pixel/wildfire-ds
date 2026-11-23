"""3-3 기상 탐색: 지속성 검사의 근거 + 튀는 값의 이웃 비교. 저장하지 않고 출력만 한다.

① 같은 값이 이어지는 길이: 같은 관측소, 정확히 1시간 간격으로 이어진 행만 (월 경계도 이어서 계산)
② 튀는 값(3-2 ③)을 가까운 관측소 3곳과 비교: 이웃도 같이 변했으면 날씨, 혼자만 튀었으면 오류 의심
③ 습도 5% 미만·일 강수 500mm 초과가 많은 관측소의 이름과 높이
"""
import numpy as np
import pandas as pd
from silver_explore_common import load_aws, load_station_info

pd.set_option("display.width", 220)
df = load_aws()
df["month"] = df.obs_time.dt.strftime("%Y%m")
cont = (df.STN == df.STN.shift()) & (df.obs_time - df.obs_time.shift() == pd.Timedelta(hours=1))


# ① 같은 값이 이어지는 길이
def runs(c, value=None):
    v = df[c]
    same_as_prev = cont & (v == v.shift()) & v.notna()
    run_id = (~same_as_prev).cumsum()
    g = pd.DataFrame({"run": run_id, "STN": df.STN, "t": df.obs_time, "v": v})[v.notna()]
    r = g.groupby("run").agg(STN=("STN", "first"), start=("t", "first"), length=("t", "size"), value=("v", "first"))
    if value is not None:
        r = r[r.value == value]
    return r


print("=== ① 같은 값이 이어지는 길이 (시간) ===")
bins = [1, 3, 6, 12, 24, 48, 10**6]
labels = ["1~2", "3~5", "6~11", "12~23", "24~47", "48+"]
for c, val, name in [("TA", None, "TA 모든 값"), ("TA", 0.0, "TA = 0.0"), ("HM", None, "HM 모든 값"), ("HM", 100.0, "HM = 100"),
                     ("WS", 0.0, "WS = 0"), ("WD", None, "WD 모든 값"), ("PS", None, "PS 모든 값"), ("PA", None, "PA 모든 값")]:
    r = runs(c, val)
    cut = pd.cut(r.length, bins=bins, labels=labels, right=False)
    tab = pd.DataFrame({"구간 수": cut.value_counts().reindex(labels),
                        "행 수": r.length.groupby(cut, observed=False).sum().reindex(labels)})
    print(f"\n[{name}]  ", "  ".join(f"{k}: {int(a):,}구간/{int(b):,}행" for k, (a, b) in tab.iterrows()))
    top = r.sort_values("length", ascending=False).head(5)
    for _, x in top.iterrows():
        print(f"     최장: STN {x.STN} {x.start:%Y-%m-%d %H:%M}부터 {x.length}시간, 값 {x.value:g}")

# ② 튀는 값의 이웃 비교
print("\n=== ② 튀는 값을 가까운 관측소 3곳과 비교 ===")
info = load_station_info()
val = df.set_index(["STN", "obs_time"])
nxt = (df.STN == df.STN.shift(-1)) & (df.obs_time.shift(-1) - df.obs_time == pd.Timedelta(hours=1))


# 관측소 × 월마다 가까운 관측소 10곳을 미리 계산해 둔다
NB = {}
for mo, m in info.groupby("month"):
    lat, lon = np.radians(m.LAT.to_numpy()), np.radians(m.LON.to_numpy())
    dx = (lon[:, None] - lon[None, :]) * np.cos(lat[:, None])
    dy = lat[:, None] - lat[None, :]
    km = 6371 * np.hypot(dx, dy)
    ids = m.STN.to_numpy()
    for i, s in enumerate(ids):
        order = np.argsort(km[i])[1:11]                       # 자기 자신(0km) 제외
        NB[(s, mo)] = list(zip(ids[order], km[i][order]))


def neighbors(stn, month):
    return NB.get((stn, month), [])


for c, t in [("TA", 5), ("HM", 30), ("PS", 5)]:
    up = (df[c] - df[c].shift()).where(cont)
    down = (df[c].shift(-1) - df[c]).where(nxt)
    spikes = df[((up > t) & (down < -t)) | ((up < -t) & (down > t))]
    if len(spikes) > 3000:
        print(f"[{c}] 튐 {len(spikes):,}행 중 앞 3,000행만 비교")
        spikes = spikes.head(3000)
    rows = []
    for i, r in spikes.iterrows():
        my_jump = float(up[i])
        nb = []
        for s, km in neighbors(r.STN, r.month):
            try:
                a = val.at[(s, r.obs_time - pd.Timedelta(hours=1)), c]
                b = val.at[(s, r.obs_time), c]
            except KeyError:
                continue
            if pd.notna(a) and pd.notna(b):
                nb.append((b - a, km))
            if len(nb) == 3:
                break
        if nb:
            rows.append({"STN": r.STN, "시각": r.obs_time, "튐": round(my_jump, 1),
                         "이웃_변화_중앙값": round(float(np.median([x for x, _ in nb])), 1),
                         "이웃_거리_km": round(float(np.mean([k for _, k in nb])), 1)})
    res = pd.DataFrame(rows)
    if res.empty:
        print(f"[{c}] 튐 > {t}: 비교할 이웃 없음")
        continue
    same_dir = (np.sign(res["튐"]) == np.sign(res["이웃_변화_중앙값"])) & (res["이웃_변화_중앙값"].abs() > t / 2)
    print(f"\n[{c}] 튐 > {t}: {len(spikes)}행 중 이웃 비교 {len(res)}행 / 이웃도 같은 방향으로 크게 변함 {int(same_dir.sum())}행 / 혼자만 튐 {int((~same_dir).sum())}행")
    print("   이웃 평균 거리 분포(km):", res["이웃_거리_km"].describe()[["min", "50%", "max"]].round(1).to_dict())
    print("   혼자만 튄 예 (튐이 큰 순 5개):")
    print(res[~same_dir].reindex(res[~same_dir]["튐"].abs().sort_values(ascending=False).index).head(5).to_string(index=False))
    print("   이웃도 같이 변한 예 (5개):")
    print(res[same_dir].head(5).to_string(index=False))

# ②-2 기온이 1시간에 15℃ 넘게 변한 행: 어느 관측소에 몰려 있나
print("\n=== ②-2 기온 1시간 변화 > 15℃ ===")
dta = (df.TA - df.TA.shift()).where(cont)
big = df[dta.abs() > 15].assign(변화=dta[dta.abs() > 15].round(1))
print(f"{len(big)}행, 관측소 {big.STN.nunique()}곳. 관측소별 상위 10:", big.STN.value_counts().head(10).to_dict())
for s_id in big.STN.value_counts().head(3).index:
    g = big[big.STN == s_id]
    print(f"   STN {s_id}: {g.obs_time.min():%Y-%m-%d} ~ {g.obs_time.max():%Y-%m-%d}, 시각(시) 분포 {g.obs_time.dt.hour.value_counts().head(3).to_dict()}")
    for i in g.index[:3]:
        w = df.loc[i - 2:i + 2]
        w = w[w.STN == s_id]
        print("      ", " → ".join(f"{t:%m-%d %H시} {v:g}" for t, v in zip(w.obs_time, w.TA)))

# ③ 특이 관측소의 이름과 높이
print("\n=== ③ 습도 5% 미만·일 강수 500mm 초과가 많은 관측소 ===")
last = info.sort_values("month").groupby("STN").tail(1).set_index("STN")
for name, ids in [("습도 5% 미만", [352, 965, 869, 868, 867, 353, 316, 485, 315, 751]),
                  ("일 강수 500mm 초과", [782, 867, 868, 869, 870, 871, 965])]:
    print(f"[{name}]")
    print(last.reindex(ids)[["STN_KO", "HT", "LAT", "LON"]].round(3).to_string())