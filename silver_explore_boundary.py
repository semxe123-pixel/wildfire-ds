"""6-3 탐색: 행정동 경계(TL_SCCO_GEMD)로 관측소 위치를 판정할 수 있는지 확인한다. 저장하지 않고 출력만 한다.

입력  data/bronze/boundary/<기준월>/*.zip (시도 17개), data/silver/stn_month.parquet, data/silver/adm_map.parquet
① 좌표계 확인: zip에 좌표계 파일(.prj)이 없음 → 후보 좌표계마다 관측소를 시군구 경계(TL_SCCO_SIG)에 넣어 보고,
   지점정보 주소의 시군구 이름과 맞는 비율로 판정
② 행정동 코드: 경계 파일의 행정동 코드와 변환표(adm_map)의 그날 유효한 행정동 코드가 같은가
③ 잘못된 도형: 몇 개이고, 고치면(make_valid) 넓이가 얼마나 바뀌나
④ 관측소 → 행정동: 경계 안에 들어가는 관측소, 어느 경계에도 안 들어가는 관측소(바닷가·섬), 두 경계에 겹치는 관측소
"""
import os
import sys
import warnings
import zipfile
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import make_valid

warnings.filterwarnings("ignore", message=".*winding order.*")     # 도형 방향 자동 교정 경고 (읽을 때 고쳐짐)
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)
B = "data/bronze/boundary"
TZ = "Asia/Seoul"
CANDIDATES = ["EPSG:5179", "EPSG:5186", "EPSG:5174"]               # UTM-K(GRS80) / 중부원점(GRS80) / 중부원점(Bessel)
MONTH_DATE = {"202411": "2024-11-30", "202512": "2025-12-31"}       # 경계 기준월 → 변환표와 비교할 날짜


def read_layer(month, layer):
    """기준월 폴더의 시도 zip 17개에서 한 층을 읽어 이어 붙임. 좌표계는 아직 정하지 않음"""
    parts = []
    folder = os.path.join(B, month)
    for z in sorted(os.listdir(folder)):
        if not z.lower().endswith(".zip"):
            continue
        path = os.path.join(folder, z)
        with zipfile.ZipFile(path) as zf:
            shp = [n for n in zf.namelist() if n.endswith(f"/{layer}.shp") or n == f"{layer}.shp"]
        for s in shp:
            g = gpd.read_file(f"zip://{path}!{s}", encoding="cp949")
            g["src_zip"] = z
            parts.append(g)
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry")


sm = pd.read_parquet("data/silver/stn_month.parquet")
adm = pd.read_parquet("data/silver/adm_map.parquet")
emd = adm[adm.admin_level == "emd"]
GEMD, SIG = {}, {}
for mo in MONTH_DATE:
    GEMD[mo] = read_layer(mo, "TL_SCCO_GEMD")
    SIG[mo] = read_layer(mo, "TL_SCCO_SIG")
    print(f"{mo}: 행정동 경계 {len(GEMD[mo]):,}행, 시군구 경계 {len(SIG[mo]):,}행, 범위 {[round(float(v)) for v in GEMD[mo].total_bounds]}")

# ================================================================ ① 좌표계
print("\n=== ① 좌표계 후보별: 관측소가 들어간 시군구 이름이 지점정보 주소와 맞는 비율 (2024-11 관측소, 2024-11 경계) ===")
st = sm[sm.month == "202411"].copy()                               # 그 달 지점정보에 있는 관측소 전부
st["addr"] = st.law_addr_raw.str.replace(" ", "", regex=False)
for crs in CANDIDATES:
    pts = gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st.lon, st.lat), crs="EPSG:4326").to_crs(crs)
    sig = SIG["202411"].set_crs(crs, allow_override=True)
    j = gpd.sjoin(pts, sig[["SIG_KOR_NM", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated()]
    inside = j.SIG_KOR_NM.notna()
    match = inside & np.array([str(n).replace(" ", "") in a for n, a in zip(j.SIG_KOR_NM, j.addr)])
    print(f"  {crs}: 관측소 {len(j)} / 어느 시군구 안에 들어감 {int(inside.sum())} / 주소의 시군구와 이름이 맞음 {int(match.sum())} ({match.mean():.1%})")
    if crs == CANDIDATES[0]:
        bad = j[inside & ~match][["station_id", "station_name", "SIG_KOR_NM", "law_addr_raw"]].head(10)
        print("    (첫 후보에서 이름이 안 맞는 예. 2026년 행정구역 이름이 적힌 주소 때문일 수 있음)")
        print("    " + bad.to_string(index=False).replace("\n", "\n    "))
CRS = CANDIDATES[0]
print(f"→ 아래는 {CRS}로 계속 (위 비율이 가장 높은 좌표계여야 함)")
for mo in MONTH_DATE:
    GEMD[mo] = GEMD[mo].set_crs(CRS, allow_override=True)

# ================================================================ ② 행정동 코드 비교
print("\n=== ② 경계 파일 행정동 코드 vs 변환표(그날 유효한 행정동) ===")
for mo, d in MONTH_DATE.items():
    g = GEMD[mo]
    codes = g.EMD_CD.astype(str).str.strip()
    dup = codes[codes.duplicated()].unique()
    day = pd.Timestamp(d, tz=TZ)
    v = emd[(emd.valid_from <= day) & (emd.valid_to.isna() | (emd.valid_to > day))]
    ref = v.drop_duplicates("admin_dong_code").set_index("admin_dong_code")
    only_g = sorted(set(codes) - set(ref.index))
    only_a = sorted(set(ref.index) - set(codes))
    print(f"[{mo} 경계 vs 변환표 {d}] 경계 {len(g):,}행 = 코드 {codes.nunique():,}개 (중복 코드 {len(dup)}개) / 변환표 행정동 {len(ref):,}개")
    print(f"   경계에만 있음 {len(only_g)}개, 변환표에만 있음 {len(only_a)}개")
    name = dict(zip(codes, g.EMD_KOR_NM))
    print("   경계에만 (예 15):", [(c, name[c]) for c in only_g[:15]])
    print("   변환표에만 (예 15):", [(c, ref.loc[c, "sigungu_name"] + " " + ref.loc[c, "admin_dong_name"]) for c in only_a[:15]])

# ================================================================ ③ 잘못된 도형
print("\n=== ③ 잘못된 도형 (경계선이 스스로 겹치는 등) ===")
for mo in MONTH_DATE:
    g = GEMD[mo]
    bad = ~g.geometry.is_valid
    fixed = make_valid(g.geometry[bad].values)
    a0, a1 = g.geometry[bad].area.values, gpd.GeoSeries(fixed, crs=CRS).area.values
    if bad.any():
        ok = a0 > 0
        ratio = a1[ok] / a0[ok]
        print(f"[{mo}] 잘못된 도형 {int(bad.sum())}개 / 원래 넓이가 0으로 계산되는 도형 {int((~ok).sum())}개 / "
              f"고친 뒤 넓이 비율(고친 뒤 ÷ 전): " + (f"최소 {ratio.min():.4f}, 최대 {ratio.max():.4f}" if ok.any() else "-"))
    else:
        print(f"[{mo}] 잘못된 도형 0개")
    if bad.any():
        print("   예:", g[bad][["EMD_CD", "EMD_KOR_NM", "src_zip"]].head(8).values.tolist())

# ================================================================ ④ 관측소 → 행정동
print("\n=== ④ 관측소 좌표가 어느 행정동 경계 안에 있나 ===")
for mo in MONTH_DATE:
    g = GEMD[mo].copy()
    g["geometry"] = make_valid(g.geometry.values)
    s = sm[(sm.month == mo) & sm.has_aws_data]                    # 기상 자료가 있는 관측소만
    pts = gpd.GeoDataFrame(s, geometry=gpd.points_from_xy(s.lon, s.lat), crs="EPSG:4326").to_crs(CRS)
    j = gpd.sjoin(pts, g[["EMD_CD", "EMD_KOR_NM", "geometry"]], how="left", predicate="within")
    n_hit = j.groupby(level=0).EMD_CD.count()
    print(f"[{mo}] 관측소 {len(pts)} / 경계 하나 안 {int((n_hit == 1).sum())} / 두 개 이상 {int((n_hit > 1).sum())} / 어느 경계에도 없음 {int((n_hit == 0).sum())}")
    out = pts[n_hit.reindex(pts.index).fillna(0).values == 0]
    if len(out):
        near = gpd.sjoin_nearest(out, g[["EMD_CD", "EMD_KOR_NM", "geometry"]], how="left", distance_col="dist_m")
        near = near[~near.index.duplicated()]
        print(f"   경계 밖 관측소 {len(near)}곳과 가장 가까운 행정동까지 거리(m), 먼 순 30:")
        print("   " + near[["station_id", "station_name", "law_addr_raw", "EMD_KOR_NM", "dist_m"]].round(0)
              .sort_values("dist_m", ascending=False).head(30).to_string(index=False).replace("\n", "\n   "))
    per = j.dropna(subset=["EMD_CD"]).groupby("EMD_CD").station_id.nunique()
    print(f"   관측소가 있는 행정동 {len(per):,}개 / 전체 {g.EMD_CD.nunique():,}개 ({len(per) / g.EMD_CD.nunique():.1%}). 한 행정동에 관측소 2곳 이상: {int((per >= 2).sum())}개")