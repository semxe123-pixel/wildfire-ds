"""3-1 기상 탐색: 물리한계 검사의 근거 만들기. 저장하지 않고 출력만 한다.

① 요소별 값 분포 (최솟값~최댓값, 극단 분위수)
② 후보 기준값마다 걸리는 건수
③ 현지기압(PA)을 관측소 높이로 계산한 예상 기압과 비교
④ 해면기압(PS) 분포, PS와 PA의 관계
"""
import numpy as np
import pandas as pd
from silver_explore_common import load_aws, load_station_height, q, NUM

pd.set_option("display.width", 200)
df = load_aws()
print(f"전체 {len(df):,}행, 관측소 {df.STN.nunique()}곳, {df.obs_time.min()} ~ {df.obs_time.max()}\n")

# ① 요소별 분포
print("=== ① 요소별 분포 (결측 표시 -99는 이미 빈 값) ===")
print(pd.DataFrame({c: q(df[c]) for c in NUM}).T.to_string())

# ② 후보 기준값마다 걸리는 건수
print("\n=== ② 후보 기준값별 건수 ===")
checks = {
    "TA < -35": df.TA < -35, "TA < -30": df.TA < -30, "TA > 40": df.TA > 40, "TA > 42": df.TA > 42,
    "HM < 0": df.HM < 0, "HM = 0": df.HM == 0, "HM < 5": df.HM < 5, "HM > 100": df.HM > 100,
    "WS < 0": df.WS < 0, "WS > 40": df.WS > 40, "WS > 60": df.WS > 60,
    "WD < 0": df.WD < 0, "WD > 360": df.WD > 360,
    "RN_HR1 < 0": df.RN_HR1 < 0, "RN_HR1 > 100": df.RN_HR1 > 100, "RN_HR1 > 150": df.RN_HR1 > 150,
    "RN_DAY < 0": df.RN_DAY < 0, "RN_DAY > 500": df.RN_DAY > 500,
    "PA = 0": df.PA == 0, "PA < 500": df.PA < 500, "PA < 800": df.PA < 800,
    "PS = 0": df.PS == 0, "PS < 900": df.PS < 900, "PS < 950": df.PS < 950, "PS > 1050": df.PS > 1050, "PS > 1060": df.PS > 1060,
}
for name, m in checks.items():
    n = int(m.sum())
    ex = ""
    if 0 < n:
        r = df[m].iloc[0]
        ex = f"  예: STN {r.STN} {r.obs_time:%Y-%m-%d %H:%M} 값 {r[name.split()[0]]}"
    print(f"  {name:<14} {n:>10,}{ex}")

# ③ PA vs 관측소 높이로 계산한 예상 기압
#    표준대기에서 높이 h(m)의 기압 ≈ 1013.25 × (1 − 0.0000225577 × h)^5.25588 hPa
print("\n=== ③ 현지기압(PA) − 높이로 계산한 예상 기압 ===")
ht = load_station_height()
p = df[df.PA.notna()][["STN", "obs_time", "PA"]].copy()
p["month"] = p.obs_time.dt.strftime("%Y%m")
p = p.merge(ht, on=["STN", "month"], how="left")
print(f"  PA 있는 행 {len(p):,} / 높이를 못 찾은 행 {p.HT.isna().sum():,}")
p["expected"] = 1013.25 * (1 - 2.25577e-5 * p.HT) ** 5.25588
p["diff"] = p.PA - p.expected
print("  차이 분포:", q(p["diff"]))
for t in (30, 50, 100, 300):
    m = p["diff"].abs() > t
    print(f"  |차이| > {t:>3} hPa: {int(m.sum()):>8,}행, 관측소 {p[m].STN.nunique()}곳")
big = p[p["diff"].abs() > 50]
if len(big):
    print("  |차이| > 50 hPa인 행이 많은 관측소 (상위 10):")
    s = big.groupby("STN").agg(행=("PA", "size"), PA_중앙값=("PA", "median"), 예상=("expected", "median"), 높이=("HT", "median"))
    print(s.sort_values("행", ascending=False).head(10).round(1).to_string())

# ④ PS 분포와 PS·PA 관계
print("\n=== ④ 해면기압(PS)과 현지기압(PA)의 관계 ===")
both = df[df.PA.notna() & df.PS.notna()]
d = both.PS - both.PA
print(f"  둘 다 있는 행 {len(both):,}")
print("  PS − PA 분포:", q(d))
print(f"  PS < PA (해면기압이 현지기압보다 낮음) {int((d < 0).sum()):,}행 / PS < PA − 5 {int((d < -5).sum()):,}행")