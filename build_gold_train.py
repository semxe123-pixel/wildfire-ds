"""Gold G3: 학습 표 → data/gold/gold_train.parquet / .csv

한 줄 = 산불 사건 1건. 그 사건이 난 행정동의 '발생 시각 정시' Feature(G2) + Label(Silver fire_label).
Feature 표(행정동 × 1시간)와 Label 표(사건)가 실제로 이어진다는 것을 보여 주는 표이자, 나중 Baseline의 입력.

규칙
  T1 기준 시각 = 발생 시각을 정시로 내림 (11:24 → 11:00). 그 줄의 Feature는 10:00 관측까지만 씀(G2 F1) → 발생 이후 정보 없음
  T2 행정동 = fire_label.admin_dong_code (발생 시점 코드). 행정동을 못 정한 사건(4건)은 빼고 수를 기록
  T3 시간 Feature 추가: hour, month, dayofweek, is_weekend (기준 시각 기준)
     바람 벡터 wind_u = ws·sin(풍향), wind_v = ws·cos(풍향). 무풍(풍향 빈 값, 풍속 0.2 m/s 미만)이면 0
     (풍향 sin·cos는 무풍일 때 빈 값이라 모델 입력에 빈칸이 생김 → 벡터는 무풍을 0으로 자연스럽게 표현)
  T4 Label: damage_area_ha(주), log_damage_area(= log(1 + ha), 치우침 완화), deaths, injuries, property_damage_kkrw,
     has_casualty(사망+부상 ≥ 1, 소방 연결 없으면 빈 값), duration_h(시각 이상이면 빈 값)
  T5 예측에 쓰면 안 되는 열(발생 뒤에야 알 수 있는 값)은 이름 앞에 'post_'를 붙임: 원인, 진화 시각, 소방 접수 정보
  T6 split: 2024 = train, 2025 = test (시간 순서 분리. 무작위 분리 금지)

입력  data/gold/feature/gold_feature_YYYYMM.parquet, data/silver/fire_label.parquet
출력  data/gold/gold_train.parquet / .csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)
OUT = "data/gold/gold_train"

fl = pd.read_parquet("data/silver/fire_label.parquet")
n_all = len(fl)
fl = fl[fl.admin_dong_code.notna()].copy()                                                  # T2
fl["forecast_reference_time"] = fl.ignition_at.dt.floor("h")                                 # T1
fl["m"] = fl.forecast_reference_time.dt.strftime("%Y%m")

parts = []
for m, g in fl.groupby("m"):
    f = f"data/gold/feature/gold_feature_{m}.parquet"
    feat = pd.read_parquet(f)
    feat = feat[feat.admin_dong_code.isin(set(g.admin_dong_code))]
    parts.append(g.merge(feat, on=["admin_dong_code", "forecast_reference_time"], how="left", indicator=True))
tr = pd.concat(parts, ignore_index=True)
no_feat = tr[tr._merge == "left_only"]
tr = tr.drop(columns=["_merge", "m"])

t = tr.forecast_reference_time                                                              # T3
tr["hour"] = t.dt.hour
tr["month"] = t.dt.month
tr["dayofweek"] = t.dt.dayofweek
tr["is_weekend"] = (tr.dayofweek >= 5).astype(int)
calm = tr.wd_sin.isna() & (tr.ws < 0.2)
tr["wind_u"] = np.where(calm, 0.0, tr.ws * tr.wd_sin)
tr["wind_v"] = np.where(calm, 0.0, tr.ws * tr.wd_cos)

tr["log_damage_area"] = np.log1p(tr.damage_area_ha)                                         # T4
tr["has_casualty"] = np.where(tr.fire119_linked, ((tr.deaths.fillna(0) + tr.injuries.fillna(0)) >= 1).astype(float), np.nan)
tr["duration_h"] = tr.duration_h.where(tr.qc_time_flag == "")
tr["split"] = np.where(tr.year == 2024, "train", "test")                                    # T6

post = {"fire_cause": "post_fire_cause", "extinguished_at": "post_extinguished_at",            # T5
        "fire119_received_at": "post_fire119_received_at", "fire119_fire_type": "post_fire119_fire_type",
        "fire119_place": "post_fire119_place"}
tr = tr.rename(columns=post)

key = ["fire_id", "year", "split", "ignition_at", "forecast_reference_time", "admin_dong_code", "admin_dong_name",
       "sigungu_name", "sido_name", "addr_mapping_method", "addr_is_estimated"]
feat_cols = ["ta", "hm", "ws", "wd_sin", "wd_cos", "pa", "ps", "rn_1h", "rn_day",
             "ta_max_24h", "ta_min_24h", "hm_min_24h", "hm_mean_24h", "ws_max_24h", "ws_mean_6h",
             "rn_sum_24h", "rn_sum_72h", "dry_hours", "eff_hm", "wind_u", "wind_v", "hour", "month", "dayofweek", "is_weekend"]
track = ["station_count", "n_check_used"] + [c for c in tr.columns if c.endswith("_dist_km") or c.endswith("_age_h")]
label = ["damage_area_ha", "log_damage_area", "deaths", "injuries", "property_damage_kkrw", "has_casualty", "duration_h",
         "fire119_linked"]
postc = list(post.values())
tr = tr[key + feat_cols + track + label + postc].sort_values("fire_id").reset_index(drop=True)

# 검사
assert tr.fire_id.is_unique
assert (tr.forecast_reference_time <= tr.ignition_at).all(), "기준 시각이 발생 시각보다 늦음"
os.makedirs("data/gold", exist_ok=True)
tr.to_parquet(OUT + ".parquet", index=False)
tr.to_csv(OUT + ".csv", index=False, encoding="utf-8-sig")

print(f"[학습 표] {len(tr)}행 × {tr.shape[1]}열 (전체 사건 {n_all}, 행정동 못 정해 뺀 사건 {n_all - len(fl)})")
print(f"  Feature 못 붙은 사건: {len(no_feat)}" + (f" → {no_feat.fire_id.tolist()}" if len(no_feat) else ""))
print("  split:", tr.split.value_counts().to_dict())
print("  Feature 빈 값(%):", tr[feat_cols].isna().mean().mul(100).round(2)[lambda x: x > 0].to_dict() or "없음")
print("  Label 빈 값(건):", tr[label].isna().sum()[lambda x: x > 0].to_dict())
print("\n[Feature 요약 (연도별 중앙값)]")
print(tr.groupby("split")[["ta", "hm", "ws", "hm_min_24h", "rn_sum_72h", "dry_hours", "eff_hm", "log_damage_area"]]
      .median().round(2).to_string())
print("\n[큰 산불 6건]")
print(tr[tr.damage_area_ha >= 100][["fire_id", "forecast_reference_time", "admin_dong_name", "ta", "hm", "ws",
                                    "hm_min_24h", "dry_hours", "eff_hm", "damage_area_ha"]].to_string(index=False))
print(f"\n[저장] {OUT}.parquet / .csv")