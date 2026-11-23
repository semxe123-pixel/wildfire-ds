"""행정동 경계 파일 확인: 받은 zip 안에 무엇이 있는지 본다. 아무것도 바꾸거나 저장하지 않는다.

입력  data/bronze/boundary/<기준월>/*.zip (행안부 주소기반산업지원서비스 "구역의 도형", 시도별 17개, 받은 그대로)
      예: data/bronze/boundary/202411/구역의도형_전체분_서울특별시.zip
출력  화면 출력만
  ① 기준월마다 zip 개수와 시도 이름 (17개 시도가 다 있나)
  ② 기준월마다 첫 zip 하나: 안의 파일 목록, 도형 파일마다 열 이름과 처음 3행
  ③ 모든 zip × 도형 파일 요약: 좌표계, 행 수, 잘못된 도형 수
"""
import os
import sys
import zipfile
import pandas as pd
import geopandas as gpd

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)
IN_DIR = "data/bronze/boundary"
SIDO = ["서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시", "대전광역시", "울산광역시", "세종특별자치시",
        "경기도", "강원특별자치도", "충청북도", "충청남도", "전북특별자치도", "전라남도", "경상북도", "경상남도", "제주특별자치도"]


def read_layer(path, shp, cpg):
    enc = cpg.get(shp[:-4]) or "cp949"                              # .cpg에 적힌 문자표, 없으면 cp949
    return gpd.read_file(f"zip://{path}!{shp}", encoding=enc), enc


months = sorted(d for d in os.listdir(IN_DIR) if os.path.isdir(os.path.join(IN_DIR, d)))
print(f"기준월 폴더: {months}")
summary = []
for mo in months:
    folder = os.path.join(IN_DIR, mo)
    zips = sorted(f for f in os.listdir(folder) if f.lower().endswith(".zip"))
    found = [s for s in SIDO if any(s in z for z in zips)]
    print(f"\n{'=' * 80}\n① {mo}: zip {len(zips)}개, 시도 이름이 맞는 것 {len(found)}개")
    missing = [s for s in SIDO if s not in found]
    if missing:
        print(f"   이름으로 찾지 못한 시도: {missing}  (이름이 다르게 적혔을 수 있음. 아래 목록 확인)")
        print(f"   zip 목록: {zips}")
    for i, z in enumerate(zips):
        path = os.path.join(folder, z)
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            cpg = {n[:-4]: zf.read(n).decode("ascii", "ignore").strip() for n in names if n.lower().endswith(".cpg")}
            if i == 0:
                print(f"\n② {mo}/{z} 안의 파일:")
                for info in zf.infolist():
                    print(f"   {info.filename:<70} {info.file_size / 1e6:>8.2f} MB")
        for shp in [n for n in names if n.lower().endswith(".shp")]:
            try:
                g, enc = read_layer(path, shp, cpg)
            except Exception as e:
                summary.append({"month": mo, "zip": z, "layer": os.path.basename(shp), "error": str(e)[:60]})
                continue
            if i == 0:
                print(f"\n   [{os.path.basename(shp)}] 문자표 {enc}, 열 {[c for c in g.columns if c != 'geometry']}")
                print("   " + g.drop(columns="geometry").head(3).to_string(index=False).replace("\n", "\n   "))
            summary.append({"month": mo, "zip": z.replace(".zip", ""), "layer": os.path.basename(shp)[:-4],
                            "crs": g.crs.to_string() if g.crs else "없음", "rows": len(g),
                            "invalid": int((~g.geometry.is_valid).sum()), "empty": int(g.geometry.is_empty.sum())})

s = pd.DataFrame(summary)
print(f"\n{'=' * 80}\n③ 모든 zip × 도형 파일 요약")
print(s.to_string(index=False))
if "rows" in s:
    print("\n기준월 × 층별 전국 합계 (행 수, 잘못된 도형 수):")
    print(s.groupby(["month", "layer"])[["rows", "invalid", "empty"]].sum().to_string())
    print("좌표계 종류:", s.crs.value_counts().to_dict())