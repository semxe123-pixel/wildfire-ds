import pandas as pd
import numpy as np

PATH = "data/bronze/aws_raw/aws_raw_202503.parquet"
NUM_COLS = ["TA", "WD", "WS", "RN_DAY", "RN_HR1", "HM", "PA", "PS"]

raw = pd.read_parquet(PATH)
print(f"행 {len(raw):,}, 열 {list(raw.columns)}")

# 1-1 자료형 변환 (탐색용 복사본에서)
df = raw.copy()
df["STN"] = df["STN"].astype(int)
for c in NUM_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")   # 숫자로 못 바꾸는 값은 NaN
print("\n[1-1] 숫자 변환 실패(원래 값이 숫자가 아님):",
      {c: int(df[c].isna().sum() - (raw[c] == "").sum()) for c in NUM_COLS})

# 1-2 시각 변환
df["obs_time"] = pd.to_datetime(df["YYMMDDHHMI"], format="%Y%m%d%H%M")
print(f"[1-2] 시각 범위 {df['obs_time'].min()} ~ {df['obs_time'].max()}, 고유 시각 {df['obs_time'].nunique()}개")

# 1-3 결측 표시값 개수 (변환 전 숫자 기준)
print("\n[1-3] -99 계열 개수:")
for c in NUM_COLS:
    n99 = int(df[c].isin([-99, -99.0, -999]).sum())
    print(f"   {c}: {n99:>8,}  ({n99 / len(df):.2%})")

# 1-4 범위: -99를 NaN으로 바꾼 뒤 열별 최소/최대/분위
for c in NUM_COLS:
    df.loc[df[c] <= -99, c] = np.nan
print("\n[1-4] 열별 최소 / 1% / 중앙 / 99% / 최대 (-99 제거 후):")
print(df[NUM_COLS].describe(percentiles=[.01, .5, .99]).T[["min", "1%", "50%", "99%", "max"]].round(1))
print("   범위 이탈 후보:",
      {"HM>100": int((df.HM > 100).sum()), "HM<0": int((df.HM < 0).sum()),
       "WS<0": int((df.WS < 0).sum()), "RN_HR1<0": int((df.RN_HR1 < 0).sum()),
       "TA<-40": int((df.TA < -40).sum()), "TA>45": int((df.TA > 45).sum()),
       "WD>360": int((df.WD > 360).sum())})

# 1-5 TA 0.0 의심값
zero = df[(df.TA == 0.0)]
zero_susp = zero[zero.PA.isna() & zero.PS.isna()]
print(f"\n[1-5] TA==0.0 행 {len(zero):,}개, 그중 PA·PS 모두 NaN {len(zero_susp):,}개")
print("   의심 행이 많은 관측소 상위 5:", zero_susp.STN.value_counts().head().to_dict())
if len(zero_susp):
    s = zero_susp.STN.value_counts().index[0]
    t = zero_susp[zero_susp.STN == s].obs_time.iloc[0]
    win = df[(df.STN == s) & (df.obs_time.between(t - pd.Timedelta(hours=3), t + pd.Timedelta(hours=3)))]
    print(f"   예시: STN {s}, {t} 전후 6시간 기온:\n", win[["obs_time", "TA", "HM", "PA"]].to_string(index=False))

# 1-6 중복
dup = df.duplicated(subset=["STN", "obs_time"]).sum()
print(f"\n[1-6] (STN, 시각) 중복 행: {dup}")

# 1-7 관측소별 결측률 (이 달 744시각 기준)
hours = df.obs_time.nunique()
per_stn = df.groupby("STN").size()
miss_rate = 1 - per_stn / hours
print(f"\n[1-7] 관측소 {len(per_stn)}곳, 이 달 시각 {hours}개")
print("   시각 결측률 분포:", miss_rate.describe(percentiles=[.5, .9, .99]).round(3).to_dict())
print("   결측률 50% 초과 관측소:", int((miss_rate > 0.5).sum()), "곳 →", miss_rate[miss_rate > 0.5].round(2).to_dict())
print("   열별 NaN 비율:", df[NUM_COLS].isna().mean().round(3).to_dict())

# 1-8 관측소 정합
stn = pd.read_parquet("data/bronze/stn_raw.parquet")
known = set(stn.STN_ID.astype(int))
unknown = set(per_stn.index) - known
print(f"\n[1-8] 관측소 목록에 없는 STN: {len(unknown)}개 → {sorted(unknown)[:20]}")
# ---- 추가 확인 ----
print(f"\n[R3] PA < 500: {int((df.PA < 500).sum())}행, PS < 500: {int((df.PS < 500).sum())}행")
print("     PA < 500인 관측소:", df[df.PA < 500].STN.value_counts().head().to_dict())

# R5: TA 0.0 중 '연속 반복' 길이 계산
d = df.sort_values(["STN", "obs_time"]).copy()
d["is_zero"] = d.TA == 0.0
# 같은 STN 안에서 is_zero 값이 바뀔 때마다 그룹 번호 증가 → 연속 구간(run) 식별
d["run"] = (d.is_zero != d.groupby("STN").is_zero.shift()).cumsum()
run_len = d[d.is_zero].groupby("run").size()
print(f"\n[R5] TA==0.0 연속 구간 {len(run_len)}개. 길이 분포:", run_len.value_counts().sort_index().to_dict())
print(f"     길이 ≥3인 구간에 속한 행: {int(run_len[run_len >= 3].sum())} / 전체 0.0 행 {int(d.is_zero.sum())}")
# 직전 시각과의 기온 차
d["ta_prev"] = d.groupby("STN").TA.shift()
z = d[d.is_zero]
print("     0.0 행의 직전 TA: NaN", int(z.ta_prev.isna().sum()), "/ |차이|≥5℃", int((z.ta_prev.abs() >= 5).sum()),
      "/ |차이|<5℃", int((z.ta_prev.abs() < 5).sum()))
print("     길이≥3 구간 예시 관측소:", d[d.is_zero & d.run.isin(run_len[run_len >= 3].index)].STN.value_counts().head().to_dict())