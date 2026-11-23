"""6-3 탐색 2: 경계 탐색에서 나온 세 가지를 확인한다. 저장하지 않고 출력만 한다.

① 출장소: 2024-11 경계에는 출장소 구역이 따로 있고 2025-12 경계에는 없음 → 출장소 구역이 어느 읍면 안에 들어 있나
② 2024-11에서 관측소가 경계 두 개에 걸친 17건: 둘 중 하나가 출장소인가
③ 운서동 → 운서1·2동: 변환표에서 언제 바뀌나
④ 달마다 변환표의 행정동(출장소 제외)을 경계 파일 두 개가 얼마나 덮나 → 어느 달에 어느 경계를 쓸지
⑤ 관측소×월을 두 경계로 각각 판정했을 때 결과가 같은가, 그 달에 유효한 코드인가
"""
import os
import sys
import warnings
import zipfile
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import make_valid

warnings.filterwarnings("ignore", message=".*winding order.*")
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)
B, TZ, CRS = "data/bronze/boundary", "Asia/Seoul", "EPSG:5179"    # 좌표계: 탐색 1에서 확인 (시군구 이름 일치 96.7%)
BRANCH = "출장소"


def read_gemd(month):
    parts = []
    folder = os.path.join(B, month)
    for z in sorted(f for f in os.listdir(folder) if f.lower().endswith(".zip")):
        path = os.path.join(folder, z)
        with zipfile.ZipFile(path) as zf:
            shp = [n for n in zf.namelist() if n.endswith("TL_SCCO_GEMD.shp")]
        for s in shp:
            parts.append(gpd.read_file(f"zip://{path}!{s}", encoding="cp949"))
    g = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=CRS)
    g["EMD_CD"] = g.EMD_CD.astype(str).str.strip()
    g["geometry"] = make_valid(g.geometry.values)                   # 탐색 1: 고친 뒤 넓이 비율 1.0000
    return g


G = {"202411": read_gemd("202411"), "202512": read_gemd("202512")}
adm = pd.read_parquet("data/silver/adm_map.parquet")
emd = adm[adm.admin_level == "emd"]                              # 법정동 관계 행 전체 (유효한 행을 먼저 고른 뒤 코드로 줄임)
sm = pd.read_parquet("data/silver/stn_month.parquet")


def valid_codes(day):
    d = pd.Timestamp(day, tz=TZ)
    v = emd[(emd.valid_from <= d) & (emd.valid_to.isna() | (emd.valid_to > d))]
    return v.drop_duplicates("admin_dong_code").set_index("admin_dong_code")[["sigungu_name", "admin_dong_name"]]


# ================================================================ ① 출장소
print("=== ① 출장소 ===")
for mo, g in G.items():
    print(f"[{mo} 경계] 이름에 '{BRANCH}'가 들어간 구역 {int(g.EMD_KOR_NM.str.contains(BRANCH).sum())}개 / 전체 {len(g):,}")
for day in ("2024-11-30", "2025-12-31"):
    v = valid_codes(day)
    print(f"[변환표 {day}] 행정동 {len(v):,}개 중 이름에 '{BRANCH}'가 들어간 것 {int(v.admin_dong_name.str.contains(BRANCH).sum())}개")
g = G["202411"]
br = g[g.EMD_KOR_NM.str.contains(BRANCH)]
rest = g[~g.EMD_KOR_NM.str.contains(BRANCH)]
inter = gpd.overlay(br[["EMD_CD", "EMD_KOR_NM", "geometry"]].rename(columns={"EMD_CD": "br_cd", "EMD_KOR_NM": "br_nm"}),
                    rest[["EMD_CD", "EMD_KOR_NM", "geometry"]], how="intersection", keep_geom_type=True)
inter["share"] = inter.area / inter.br_cd.map(br.set_index("EMD_CD").area)
best = inter.sort_values("share", ascending=False).drop_duplicates("br_cd")
print(f"2024-11 출장소 구역 {len(br)}개 중 다른 행정동 안에 90% 이상 들어가는 것 {int((best.share >= 0.9).sum())}개 / "
      f"겹치는 행정동이 없는 것 {len(br) - best.br_cd.nunique()}개")
print("   출장소 → 그것을 품은 행정동 (예 10):", best.head(10)[["br_nm", "EMD_KOR_NM", "share"]].round(3).values.tolist())
v25 = valid_codes("2025-12-31")
only_adm = v25.loc[sorted(set(v25.index) - set(G["202512"].EMD_CD))]
print(f"[2025-12] 변환표에만 있는 {len(only_adm)}개 중 출장소 {int(only_adm.admin_dong_name.str.contains(BRANCH).sum())}개. 출장소가 아닌 것:",
      only_adm[~only_adm.admin_dong_name.str.contains(BRANCH)].reset_index().head(20).values.tolist())

# ================================================================ ② 경계 두 개에 걸친 관측소
print("\n=== ② 2024-11 경계에서 관측소가 경계 두 개 이상에 들어간 경우 ===")
s = sm[(sm.month == "202411") & sm.has_aws_data]
pts = gpd.GeoDataFrame(s, geometry=gpd.points_from_xy(s.lon, s.lat), crs="EPSG:4326").to_crs(CRS)
j = gpd.sjoin(pts, G["202411"][["EMD_CD", "EMD_KOR_NM", "geometry"]], predicate="within")
multi = j[j.index.duplicated(keep=False)]
both = multi.groupby("station_id").EMD_KOR_NM.apply(list)
print(f"{both.size}곳. 둘 중 하나가 출장소인 곳 {int(both.apply(lambda x: any(BRANCH in n for n in x)).sum())}곳")
print("   예:", both.head(8).to_dict())

# ================================================================ ③ 운서동
print("\n=== ③ 운서동 관련 변환표 행 ===")
w = adm[adm.admin_dong_name.astype(str).str.contains("운서") & (adm.admin_level == "emd")]
print(w.drop_duplicates(["admin_dong_code", "valid_from", "valid_to"])[["admin_dong_code", "admin_dong_name", "valid_from", "valid_to"]].to_string(index=False))

# ================================================================ ④ 달마다 경계가 행정동을 얼마나 덮나
print("\n=== ④ 달마다 변환표 행정동(출장소 제외)이 경계 파일에 있는가 ===")
codes = {mo: set(g.loc[~g.EMD_KOR_NM.str.contains(BRANCH), "EMD_CD"]) for mo, g in G.items()}
rows = []
for mo in pd.period_range("2024-01", "2025-12", freq="M").strftime("%Y%m"):
    v = valid_codes(mo[:4] + "-" + mo[4:] + "-01")
    v = v[~v.admin_dong_name.str.contains(BRANCH)]
    r = {"month": mo, "행정동": len(v)}
    for b, c in codes.items():
        miss = sorted(set(v.index) - c)
        r[f"{b} 경계에 없음"] = len(miss)
        r[f"{b} 예"] = [v.loc[x, "admin_dong_name"] for x in miss[:3]]
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))

# ================================================================ ⑤ 관측소×월: 두 경계 판정 비교
print("\n=== ⑤ 관측소×월을 두 경계(출장소 제외)로 판정 ===")
s = sm[sm.has_aws_data].copy()
pts = gpd.GeoDataFrame(s, geometry=gpd.points_from_xy(s.lon, s.lat), crs="EPSG:4326").to_crs(CRS)
uniq = pts.drop_duplicates(["lon", "lat"])[["lon", "lat", "geometry"]]   # 같은 좌표는 한 번만 판정
for b, g in G.items():
    gg = g[~g.EMD_KOR_NM.str.contains(BRANCH)][["EMD_CD", "geometry"]]
    hit = gpd.sjoin(uniq, gg, predicate="within")
    hit = hit[~hit.index.duplicated()]
    miss = uniq.drop(hit.index)
    near = gpd.sjoin_nearest(miss, gg, distance_col="d")
    near = near[~near.index.duplicated()]
    code = pd.concat([hit.EMD_CD, near.EMD_CD])
    key = dict(zip(zip(uniq.lon, uniq.lat), code.reindex(uniq.index)))
    s[f"code_{b}"] = [key[(x, y)] for x, y in zip(s.lon, s.lat)]
same = s.code_202411 == s.code_202512
valid_on = {}
for mo in s.month.unique():
    valid_on[mo] = set(valid_codes(mo[:4] + "-" + mo[4:] + "-01").index)
s["ok_2411"] = [c in valid_on[m] for c, m in zip(s.code_202411, s.month)]
s["ok_2512"] = [c in valid_on[m] for c, m in zip(s.code_202512, s.month)]
print(f"관측소×월 {len(s):,} / 두 경계 결과가 같음 {int(same.sum()):,} / 다름 {int((~same).sum()):,}")
print("결과가 다른 관측소×월: 각 결과가 그 달 1일에 유효한 코드인가")
print(s[~same].groupby(["ok_2411", "ok_2512"]).size().rename("관측소×월").to_string())
diff = s[~same].drop_duplicates("station_id")[["station_id", "station_name", "month", "code_202411", "code_202512", "ok_2411", "ok_2512"]]
print(f"   결과가 다른 관측소 {diff.station_id.nunique()}곳 (예 15):")
print("   " + diff.head(15).to_string(index=False).replace("\n", "\n   "))