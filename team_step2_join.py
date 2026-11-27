"""공통 학습 표 Step 2: 우리 학습 표에 완길님 열 4개 붙이기 → data/gold/team/joined.parquet

붙이는 열 (완길님 이름 → 우리 이름)
  wind_gust_max_1h_ms          → gust_max_1h   지난 1시간 최대 순간풍속 (m/s)
  wind_gust_max_6h_ms          → gust_max_6h   지난 6시간 최대 순간풍속 (m/s)
  relative_humidity_min_6h_pct → hm_min_6h     지난 6시간 최저 습도 (%)
  rainfall_7d_mm               → rn_sum_7d     지난 7일 강수 (mm)
결과 한 행 = 우리 산불 1건 (734행). 빈 값은 아직 그대로 (Step 3에서 채움)
실행: python team_step2_join.py   (wildfire 폴더에서, Step 1 다음)
"""
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
OURS = "data/gold/gold_train.parquet"
TEAM = "../wildfire_team/제출골드 3조/data/gold/damage_version/train_event_samples_damage.parquet"
MATCH = "data/gold/team/match.parquet"
OUT = "data/gold/team/joined.parquet"
ADD = {"wind_gust_max_1h_ms": "gust_max_1h", "wind_gust_max_6h_ms": "gust_max_6h",
       "relative_humidity_min_6h_pct": "hm_min_6h", "rainfall_7d_mm": "rn_sum_7d"}

o = pd.read_parquet(OURS)
mt = pd.read_parquet(MATCH)
t = pd.read_parquet(TEAM, columns=["event_id", "forecast_reference_time", "latest_observed_at"] + list(ADD))

# 1. 완길님 열이 같은 시점 정보인지: 마지막 관측 = 기준 시각 - 1시간 (누출 없음)
lag = (pd.to_datetime(t.forecast_reference_time, utc=True) - pd.to_datetime(t.latest_observed_at, utc=True))
assert (lag == pd.Timedelta(hours=1)).all(), "완길님 표에 기준 시각 1시간 전이 아닌 관측이 있음"

# 2. 짝 표로 event_id를 붙이고, 그 event_id로 완길님 열 4개를 가져옴
t = t.rename(columns=ADD)[["event_id", "forecast_reference_time"] + list(ADD.values())]
j = (o.merge(mt[["fire_id", "event_id", "team_dong_mismatch"]], on="fire_id", how="left", validate="one_to_one")
      .merge(t.rename(columns={"forecast_reference_time": "team_ref"}), on="event_id", how="left", validate="one_to_one"))

# 3. 두 표의 기준 시각이 같은지 (같은 산불이면 같아야 함)
same_ref = pd.to_datetime(j.forecast_reference_time, utc=True) == pd.to_datetime(j.team_ref, utc=True)
assert same_ref.all(), f"기준 시각이 다른 산불 {int((~same_ref).sum())}건"
j = j.drop(columns=["event_id", "team_ref"])
for c in ADD.values():
    j[c] = j[c].astype("float32")

# 4. 확인과 저장
print(f"행: {len(j)} (우리 {len(o)}) / 열: {j.shape[1]} (우리 {o.shape[1]} + 4 + 표시 열 1)")
print("붙인 열 빈 값:", j[list(ADD.values())].isna().sum().to_dict())
print(j[list(ADD.values())].describe().loc[["min", "50%", "max"]].round(1).to_string())
j.to_parquet(OUT, index=False)
print(f"저장: {OUT}")