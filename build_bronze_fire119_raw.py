"""Bronze: 소방청 화재 현황 구조화 표 (원천 레코드 표)

입력  data/bronze/fire119/*.csv              소방안전 빅데이터 플랫폼 "화재 현황" 2024·2025 (UTF-8 BOM, 60열)
      data/bronze/fire119_index.parquet      manifest 코드가 만든 파일별 행 수(n_rows) → 건수 일치 기준
출력  data/bronze/fire119_raw.parquet        화재 1건 = 1행

규칙 (Bronze: 값은 바꾸지 않음)
- 모든 값을 글자 그대로 읽음. 빈칸도 NaN으로 바꾸지 않고 빈 글자로 둠 (keep_default_na=False)
- 산불만 고르거나 단위를 바꾸는 일은 하지 않음 → Silver(build_silver_fire_label.py)에서 함
- 출처 칸: bronze_file (manifest와 같은 형식의 원본 경로)
"""
import glob
import os
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
SRC = "data/bronze/fire119"
INDEX = "data/bronze/fire119_index.parquet"
OUT = "data/bronze/fire119_raw.parquet"

frames, per_file = [], []
for path in sorted(glob.glob(os.path.join(SRC, "*.csv"))):
    name = os.path.basename(path)
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df["bronze_file"] = f"fire119/{name}"
    frames.append(df)
    per_file.append({"file": name, "n_read": len(df), "n_cols": df.shape[1] - 1})

raw = pd.concat(frames, ignore_index=True).astype("string")
raw.to_parquet(OUT, index=False)
print(f"[저장] {OUT}  {len(raw):,}행 × {raw.shape[1]}열")
print(pd.DataFrame(per_file).to_string(index=False))
same_cols = len({tuple(f.columns) for f in frames}) == 1
print("두 파일의 열 구성 같음:", same_cols)

# 건수 일치: 파일별로 읽은 행 수 = 인덱스(n_rows)
print("\n=== 건수 일치 ===")
pf = pd.DataFrame(per_file)
if os.path.exists(INDEX):
    idx = pd.read_parquet(INDEX)[["file", "n_rows"]]
    m = pf.merge(idx, on="file", how="outer", indicator=True)
    diff = m[(m["_merge"] != "both") | (m["n_read"] != m["n_rows"])]
    print(f"파일 {len(pf)}개 / 읽은 행 {pf.n_read.sum():,} / 인덱스 합계 {int(idx.n_rows.sum()):,} / 다른 파일 {len(diff)}")
    print("일치" if diff.empty and len(raw) == int(idx.n_rows.sum()) else "불일치 — 확인 필요")
    if not diff.empty:
        print(diff.to_string(index=False))
else:
    print("fire119_index.parquet 없음 → 먼저 python build_bronze_manifest.py 실행")

# 미리보기 (값은 바꾸지 않고 보기만)
print("\n[화재 유형 상위 5]")
print(raw["fire_type_nm"].value_counts().head().to_string())
print("\n임야 중 장소 중분류:", raw.loc[raw.fire_type_nm == "임야", "fclt_plc_mclsf_nm"].value_counts().to_dict())