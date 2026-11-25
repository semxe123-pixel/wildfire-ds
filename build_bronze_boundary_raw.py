"""Bronze: 행정동 경계 구조화 표 (원천 레코드 표)

입력  data/bronze/boundary/<기준월>/*.zip   행안부 "구역의 도형" (2024-11판, 2025-12판 × 시도 17)
      data/bronze/boundary_index.parquet     manifest 코드가 만든 zip별 행정동 구역 수(n_gemd) → 건수 일치 기준
출력  data/bronze/boundary_raw.parquet       행정동 구역 1개 = 1행 (GeoParquet)

규칙 (Bronze: 값은 바꾸지 않음)
- zip 안의 행정동 층 TL_SCCO_GEMD만 읽음. Silver가 쓰는 층이 이것뿐임 (다른 층은 zip에 그대로 있음)
- 속성 값은 전부 글자 그대로. 경계 모양(geometry)도 그대로 두고 고치지 않음 → 잘못된 모양 고치기(B2)는 Silver
- 좌표계 정보(.prj)가 원본에 없으므로 여기서도 정하지 않음 → EPSG:5179로 보는 것(B1)은 Silver
- 출처 칸: version(기준월), bronze_file(manifest와 같은 형식의 원본 경로), layer
"""
import os
import sys
import zipfile

import geopandas as gpd
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
BASE = "data/bronze/boundary"
INDEX = "data/bronze/boundary_index.parquet"
OUT = "data/bronze/boundary_raw.parquet"
LAYER = "TL_SCCO_GEMD"

parts, per_file = [], []
for ver in sorted(d for d in os.listdir(BASE) if os.path.isdir(os.path.join(BASE, d))):
    folder = os.path.join(BASE, ver)
    for z in sorted(f for f in os.listdir(folder) if f.lower().endswith(".zip")):
        path = os.path.join(folder, z)
        with zipfile.ZipFile(path) as zf:
            shps = [n for n in zf.namelist() if n.endswith(f"{LAYER}.shp")]
        n = 0
        for s in shps:
            g = gpd.read_file(f"zip://{path}!{s}", encoding="cp949")
            attr = [c for c in g.columns if c != "geometry"]
            g[attr] = g[attr].astype("string")                      # 속성은 글자 그대로
            g["version"] = ver
            g["bronze_file"] = f"boundary/{ver}/{z}"
            g["layer"] = LAYER
            parts.append(g)
            n += len(g)
        per_file.append({"file": f"{ver}/{z}", "n_read": n})

raw = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry")
raw.to_parquet(OUT, index=False)
print(f"[저장] {OUT}  {len(raw):,}행 × {raw.shape[1]}열")
print("열:", [c for c in raw.columns])
print("\n[기준월별 구역 수]")
print(raw.groupby("version").size().to_string())
print("\n잘못된 모양(고치지 않고 그대로 둠, Silver B2에서 고침):", int((~raw.geometry.is_valid).sum()))

# 건수 일치: zip별로 읽은 구역 수 = 인덱스(n_gemd)
print("\n=== 건수 일치 ===")
pf = pd.DataFrame(per_file)
if os.path.exists(INDEX):
    idx = pd.read_parquet(INDEX)[["file", "n_gemd"]]
    m = pf.merge(idx, on="file", how="outer", indicator=True)
    diff = m[(m["_merge"] != "both") | (m["n_read"] != m["n_gemd"])]
    print(f"zip {len(pf)}개 / 읽은 구역 {pf.n_read.sum():,} / 인덱스 합계 {int(idx.n_gemd.sum()):,} / 다른 zip {len(diff)}")
    print("일치" if diff.empty and len(raw) == int(idx.n_gemd.sum()) else "불일치 — 확인 필요")
    if not diff.empty:
        print(diff.to_string(index=False))
else:
    print("boundary_index.parquet 없음 → 먼저 python build_bronze_manifest.py 실행")