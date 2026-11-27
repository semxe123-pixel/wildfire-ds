"""공통 학습 표 Step 1: 우리 산불(fire_id)과 완길님 산불(event_id) 짝짓기 → data/gold/team/match.parquet

짝짓는 열쇠 = 발생 시각 + 피해면적(소수 둘째 자리)
  두 표의 사건 ID가 서로 달라서, 두 표에 모두 있는 값으로 같은 산불을 찾음
결과 한 행 = 우리 산불 1건 (734행). 행정동이 다르면 표시
실행: python team_step1_match.py   (wildfire 폴더에서)
"""
import os
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
TZ = "Asia/Seoul"
OURS = "data/gold/gold_train.parquet"
TEAM = "../wildfire_team/제출골드 3조/data/gold/damage_version/train_event_samples_damage.parquet"
OUT = "data/gold/team/match.parquet"

o = pd.read_parquet(OURS, columns=["fire_id", "ignition_at", "admin_dong_code", "admin_dong_name", "damage_area_ha"])
t = pd.read_parquet(TEAM, columns=["event_id", "ignition_at", "admin_dong_code", "target_value"])

# 1. 열쇠 만들기: 시각은 한국 시간으로 맞추고, 면적은 소수 둘째 자리로 반올림
o["key_t"] = pd.to_datetime(o.ignition_at).dt.tz_convert(TZ)
t["key_t"] = pd.to_datetime(t.ignition_at, utc=True).dt.tz_convert(TZ)
o["key_a"] = o.damage_area_ha.astype(float).round(2)
t["key_a"] = t.target_value.astype(float).round(2)

# 2. 열쇠가 한 표 안에서 겹치지 않는지 (겹치면 짝이 둘 이상 나옴)
for name, d in (("우리", o), ("완길님", t)):
    dup = int(d.duplicated(["key_t", "key_a"]).sum())
    assert dup == 0, f"{name} 표에 같은 열쇠가 {dup}건 있음"

# 3. 우리 표를 기준으로 붙이기
m = o.merge(t[["event_id", "key_t", "key_a", "admin_dong_code"]].rename(columns={"admin_dong_code": "team_dong_code"}),
            on=["key_t", "key_a"], how="left", validate="one_to_one")
m["admin_dong_code"] = m.admin_dong_code.astype(str)
m["team_dong_code"] = m.team_dong_code.astype("string")
m["team_dong_mismatch"] = m.team_dong_code.notna() & (m.admin_dong_code != m.team_dong_code)
jb = m.team_dong_mismatch & m.admin_dong_code.str.startswith("52") & m.team_dong_code.str.startswith("45") \
     & (m.admin_dong_code.str[2:] == m.team_dong_code.str[2:])

# 4. 검사와 저장
print(f"우리 {len(o)}건 / 완길님 {len(t)}건")
print(f"짝 찾음: {int(m.event_id.notna().sum())}건, 못 찾음: {int(m.event_id.isna().sum())}건")
print(f"행정동 다름: {int(m.team_dong_mismatch.sum())}건 (전북 코드만 다름 {int(jb.sum())}, 다른 동 {int((m.team_dong_mismatch & ~jb).sum())})")
print(f"완길님에게만 있는 산불: {len(t) - int(m.event_id.notna().sum())}건")
assert m.event_id.notna().all(), "짝을 못 찾은 우리 산불이 있음"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
m[["fire_id", "event_id", "admin_dong_code", "team_dong_code", "team_dong_mismatch"]].to_parquet(OUT, index=False)
print(f"저장: {OUT}")