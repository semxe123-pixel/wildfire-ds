"""공통 학습 표 Step 4: 분할 표시 + 두 표 저장
  전체 표   data/gold/gold_train_team.parquet / .csv   734행 × 72열. 빈 값은 원래대로 (채운 기록은 team/fill_log.csv)
  모델용 표 data/gold/model_input.parquet / .csv       734행 × 27열 = fire_id, split, 입력 24열, log_damage_area. 빈 값 없음
  입력 열 목록 data/gold/feature_columns.json
분할 (모두 고정, 발생 시각 기준)
  train ~ 2025-03-14 / valid 2025-03-15 – 2025-04-11 / test 2025-04-12 ~
실행: python team_step4_save.py   (wildfire 폴더에서, Step 3 다음)
"""
import json
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
TZ = "Asia/Seoul"
JOINED, FILLED, LOG = "data/gold/team/joined.parquet", "data/gold/team/filled.parquet", "data/gold/team/fill_log.csv"
FULL, MODEL, COLS = "data/gold/gold_train_team", "data/gold/model_input", "data/gold/feature_columns.json"
VALID_START, TEST_START = pd.Timestamp("2025-03-15", tz=TZ), pd.Timestamp("2025-04-12", tz=TZ)
FEATURES = ["ta", "hm", "ws", "wind_u", "wind_v", "pa", "ps", "gust_max_1h",
            "hm_min_6h", "hm_min_24h", "hm_mean_24h", "eff_hm", "ta_max_24h", "ta_min_24h",
            "ws_mean_6h", "ws_max_24h", "gust_max_6h", "rn_sum_72h", "rn_sum_7d", "dry_hours",
            "hour", "month", "dayofweek", "is_weekend"]
LABEL = "log_damage_area"

full = pd.read_parquet(JOINED)
filled = pd.read_parquet(FILLED)

# 1. 분할 다시 만들기 (Gold의 2024/2025 분할을 바꿈)
ign = pd.to_datetime(full.ignition_at).dt.tz_convert(TZ)
split = np.select([ign < VALID_START, ign < TEST_START], ["train", "valid"], "test")
full["split"] = split
filled["split"] = split

# 2. 검사
assert len(full) == len(filled) == 734 and full.fire_id.is_unique, "행 수나 fire_id가 이상함"
assert (full.fire_id.to_numpy() == filled.fire_id.to_numpy()).all(), "두 표의 순서가 다름"
same = full[FEATURES].isna() | np.isclose(full[FEATURES].astype(float), filled[FEATURES].astype(float))
assert same.all().all(), "원래 값이 있던 칸이 바뀜"                                   # Step 3은 빈 칸만 채워야 함
assert filled[FEATURES].notna().all().all(), "모델용 입력에 빈 값이 있음"
assert (pd.to_datetime(full.forecast_reference_time) <= pd.to_datetime(full.ignition_at)).all(), "기준 시각이 발생보다 늦음"
assert full[LABEL].notna().all(), "Label 빈 값"

# 3. 저장
full.to_parquet(FULL + ".parquet", index=False)
full.to_csv(FULL + ".csv", index=False, encoding="utf-8-sig")
model = filled[["fire_id", "split"] + FEATURES + [LABEL]]
model.to_parquet(MODEL + ".parquet", index=False)
model.to_csv(MODEL + ".csv", index=False, encoding="utf-8-sig")
n_fill = len(pd.read_csv(LOG))
with open(COLS, "w", encoding="utf-8") as f:
    json.dump({"id": "fire_id", "split": "split", "label": LABEL,
               "label_note": "log(1 + 피해면적 ha). 원래 면적 = exp(값) - 1",
               "features": FEATURES,
               "split_rule": {"train": "발생 ~ 2025-03-14", "valid": "2025-03-15 – 2025-04-11", "test": "2025-04-12 ~"},
               "rules": ["설정값·Feature 고르기는 valid로만. test는 마지막에 한 번만",
                         "최종 모델은 train + valid로 다시 학습한 뒤 test를 예측",
                         "입력은 features 24열 그대로 (주 비교)",
                         f"빈 값은 미리 채움: {n_fill}칸, 방법은 team/fill_log.csv"]},
              f, ensure_ascii=False, indent=2)

# 4. 요약
a = full.damage_area_ha.astype(float)
print(f"전체 표: {full.shape[0]}행 × {full.shape[1]}열 → {FULL}.parquet / .csv")
print(f"모델용 표: {model.shape[0]}행 × {model.shape[1]}열 → {MODEL}.parquet / .csv")
print(f"입력 열 목록: {COLS} ({len(FEATURES)}열)")
s = pd.DataFrame({"split": split, "big": a >= 100, "mid": (a >= 1) & (a < 100)})
print(s.groupby("split").agg(건수=("big", "size"), 큰산불_100ha이상=("big", "sum"), 중간_1_100ha=("mid", "sum"))
      .reindex(["train", "valid", "test"]).to_string())
print("전체 표 입력 열 빈 값(원래대로):", int(full[FEATURES].isna().sum().sum()), "/ 모델용 표 빈 값:", int(model[FEATURES].isna().sum().sum()))
print("행정동 다름 표시(전체 표에만):", int(full.team_dong_mismatch.sum()))