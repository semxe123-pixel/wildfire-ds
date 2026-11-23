"""3-2 기상 탐색: 단계검사의 근거 + 3-1에서 나온 확인거리. 저장하지 않고 출력만 한다.

① 3-1 확인거리: 높이를 못 찾은 행, STN 442의 PA 500.0, 습도 0%, 일 강수 500mm 초과
② 1시간 변화량: 바로 앞 시각(정확히 1시간 전)과의 차이 분포와 후보 기준값별 건수
③ 튀는 값: 올라갔다 바로 내려오는(또는 반대) 값의 건수와 예시
"""
import numpy as np
import pandas as pd
from silver_explore_common import load_aws, load_station_height, q

pd.set_option("display.width", 200)
df = load_aws()

# ① 3-1 확인거리
print("=== ① 3-1 확인거리 ===")
ht = load_station_height()
df["month"] = df.obs_time.dt.strftime("%Y%m")
miss = df[["STN", "month"]].drop_duplicates().merge(ht, on=["STN", "month"], how="left")
miss = miss[miss.HT.isna()]
cnt = df.merge(miss[["STN", "month"]], on=["STN", "month"]).groupby(["STN", "month"]).size()
print(f"[높이 없음] 관측소×월 {len(miss)}개, {int(cnt.sum()):,}행 (그 달 1일 지점정보에 없는 관측소)")
print(cnt.rename("행").reset_index().to_string(index=False))

s442 = df[df.STN == 442]
m500 = s442.PA == 500
print(f"\n[STN 442] 전체 {len(s442):,}행 / PA=500.0 {int(m500.sum()):,}행 / PA 빈 값 {int(s442.PA.isna().sum()):,}행 / PS 빈 값 {int(s442.PS.isna().sum()):,}행")
if m500.any():
    t = s442[m500].obs_time
    print(f"   PA=500.0 기간: {t.min():%Y-%m-%d %H:%M} ~ {t.max():%Y-%m-%d %H:%M}")
    print("   PA 값 종류 (상위 5):", s442.PA.value_counts().head(5).to_dict())

h0 = df[df.HM == 0][["STN", "obs_time", "HM", "TA"]]
print(f"\n[습도 0%] {len(h0)}행, 관측소 {h0.STN.nunique()}곳")
print(h0.to_string(index=False))
h5 = df[(df.HM > 0) & (df.HM < 5)]
print(f"[습도 0~5% 사이] {len(h5)}행, 관측소별 상위 10:", h5.STN.value_counts().head(10).to_dict())

r500 = df[df.RN_DAY > 500]
days = r500.assign(day=r500.obs_time.dt.strftime("%Y-%m-%d")).groupby(["STN", "day"]).RN_DAY.agg(["size", "max"])
print(f"\n[일 강수 > 500mm] {len(r500)}행 = 관측소×날짜 {len(days)}개")
print(days.rename(columns={"size": "행", "max": "최대"}).to_string())

# ② 1시간 변화량: 같은 관측소에서 바로 앞 행이 정확히 1시간 전일 때만
print("\n=== ② 1시간 변화량 (바로 앞 시각이 정확히 1시간 전인 경우만) ===")
same = (df.STN == df.STN.shift()) & (df.obs_time - df.obs_time.shift() == pd.Timedelta(hours=1))
print(f"비교 가능한 행 {int(same.sum()):,} / 전체 {len(df):,}")
cands = {"TA": (5, 8, 10, 15), "HM": (30, 50, 70), "PS": (5, 10, 20), "PA": (5, 10, 20), "WS": (10, 15, 20)}
for c, ts in cands.items():
    d = (df[c] - df[c].shift()).where(same).abs()
    print(f"\n{c} |변화| 분포:", q(d, ps=(0.5, 0.99, 0.999, 0.9999, 1)))
    for t in ts:
        m = d > t
        print(f"   |변화| > {t:>3}: {int(m.sum()):>8,}행, 관측소 {df[m].STN.nunique()}곳")

# ③ 튀는 값: 앞 시각 대비 +X 이상 뛰고, 다음 시각에 −X 이상 돌아옴 (또는 반대)
print("\n=== ③ 튀는 값 (한 시각만 튀었다 돌아옴) ===")
nxt = (df.STN == df.STN.shift(-1)) & (df.obs_time.shift(-1) - df.obs_time == pd.Timedelta(hours=1))
for c, ts in {"TA": (5, 8, 10), "HM": (30, 50), "PS": (5, 10)}.items():
    up = (df[c] - df[c].shift()).where(same)
    down = (df[c].shift(-1) - df[c]).where(nxt)
    for t in ts:
        m = ((up > t) & (down < -t)) | ((up < -t) & (down > t))
        print(f"   {c} 튐 > {t:>2}: {int(m.sum()):>6,}행, 관측소 {df[m].STN.nunique()}곳")
    m = ((up > ts[0]) & (down < -ts[0])) | ((up < -ts[0]) & (down > ts[0]))
    if m.any():
        idx = df.index[m][:5]
        for i in idx:
            w = df.loc[i - 1:i + 1, ["STN", "obs_time", c]]
            vals = " → ".join(f"{v:g}" for v in w[c])
            print(f"      예: STN {df.at[i, 'STN']} {df.at[i, 'obs_time']:%Y-%m-%d %H:%M}  {vals}")