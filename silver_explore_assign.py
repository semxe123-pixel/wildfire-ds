"""6-4 탐색: 행정동 대표점에서 '쓸 수 있는 관측소'까지 얼마나 먼가 (저장 없음, 출력만)

입력  data/bronze/boundary/<기준월>/*.zip   행정동 경계 (6-3과 같은 방식으로 읽음: 5179, make_valid, 출장소 제외)
      data/silver/stn_adm.parquet           관측소 × 날짜 범위 (좌표, 경계 밖 여부)
      data/silver/aws_availability.parquet  관측소 × 월 × 요소별 값이 남은 비율 (value_share)
출력  화면 출력만

보는 것
  ① 대표점: 무게중심이 구역 밖에 떨어지는 행정동 수 (그런 곳은 구역 안의 점으로 바꿈)
  ② 요소별로 그 달 쓸 수 있는 관측소 수 (value_share 기준 두 가지)
  ③ 행정동 × 월 × 요소별 가장 가까운 관측소 1·2·3번째 거리 분포
  ④ 반경 10·20·30 km 안 관측소 수
  ⑤ 가장 먼 행정동, 바다 관측소가 가장 가까운 경우
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
pd.set_option("display.max_rows", 200)

CRS, B, BRANCH = "EPSG:5179", "data/bronze/boundary", "출장소"
ELEMS = ["TA", "WD", "WS", "RN_DAY", "RN_HR1", "HM", "PA", "PS"]
SHARE = 0.5                     # 이 달 시간의 절반 이상 값이 남은 관측소를 '쓸 수 있음'으로 봄 (비교용으로 >0도 출력)
RADII = [10, 20, 30]


def version_of(month):
    return "202411" if month <= "202506" else "202512"


# ================================================================ 1. 행정동 대표점
def read_gemd(ver):
    parts = []
    folder = os.path.join(B, ver)
    for z in sorted(f for f in os.listdir(folder) if f.lower().endswith(".zip")):
        path = os.path.join(folder, z)
        with zipfile.ZipFile(path) as zf:
            shp = [n for n in zf.namelist() if n.endswith("TL_SCCO_GEMD.shp")]
        for s in shp:
            parts.append(gpd.read_file(f"zip://{path}!{s}", encoding="cp949"))
    g = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=CRS)
    g["EMD_CD"] = g.EMD_CD.astype(str).str.strip()
    g["geometry"] = make_valid(g.geometry.values)
    return g[~g.EMD_KOR_NM.str.contains(BRANCH)].reset_index(drop=True)


DONG = {}
print("[① 대표점]")
for ver in ("202411", "202512"):
    g = read_gemd(ver)
    cen = g.geometry.centroid
    inside = cen.within(g.geometry).values
    rep = g.geometry.representative_point()
    pt = gpd.GeoSeries(np.where(inside, cen.values, rep.values), crs=CRS)
    DONG[ver] = pd.DataFrame({"code": g.EMD_CD, "name": g.EMD_KOR_NM, "x": pt.x.values, "y": pt.y.values,
                              "area_km2": (g.geometry.area / 1e6).round(2)})
    out = g.loc[~inside, "EMD_KOR_NM"].tolist()
    print(f"  {ver}: 행정동 {len(g):,} / 무게중심이 구역 밖 {len(out)}곳 → 구역 안의 점 사용 / 예: {out[:8]}")

# ================================================================ 2. 관측소 좌표 (관측소 × 월)
sa = pd.read_parquet("data/silver/stn_adm.parquet")
st = sa.drop_duplicates(["station_id", "month"])[["station_id", "month", "lon", "lat", "spatial_mapping_method"]].copy()
xy = gpd.GeoSeries(gpd.points_from_xy(st.lon, st.lat), crs="EPSG:4326").to_crs(CRS)
st["x"], st["y"] = xy.x.values, xy.y.values
SEA = set(sa.loc[sa.spatial_mapping_method != "within", "station_id"])          # 구역 밖 관측소

av = pd.read_parquet("data/silver/aws_availability.parquet")
months = sorted(av.month.unique())

print("\n[② 그 달 쓸 수 있는 관측소 수 (월 평균)]")
cnt = pd.DataFrame({
    "값이 조금이라도": av[av.value_share > 0].groupby(["element", "month"]).size().groupby("element").mean(),
    f"값 {SHARE:.0%} 이상": av[av.value_share >= SHARE].groupby(["element", "month"]).size().groupby("element").mean(),
}).reindex(ELEMS).round(0)
print(cnt.to_string())

# ================================================================ 3. 행정동 × 월 × 요소: 가까운 관측소 거리
rows = []
for month in months:
    D = DONG[version_of(month)]
    s_m = st[st.month == month]
    for el in ELEMS:
        ok = av[(av.month == month) & (av.element == el) & (av.value_share >= SHARE)].station_id
        s = s_m[s_m.station_id.isin(ok)]
        if s.empty:
            continue
        d = np.hypot(D.x.values[:, None] - s.x.values[None, :], D.y.values[:, None] - s.y.values[None, :]) / 1000
        k = min(3, d.shape[1])
        order = np.argsort(d, axis=1)[:, :k]
        near = np.take_along_axis(d, order, axis=1)
        r = pd.DataFrame({"code": D.code.values, "name": D.name.values, "month": month, "element": el,
                          "d1": near[:, 0],
                          "d2": near[:, 1] if k > 1 else np.nan,
                          "d3": near[:, 2] if k > 2 else np.nan,
                          "nearest_stn": s.station_id.values[order[:, 0]]})
        for R in RADII:
            r[f"n{R}"] = (d <= R).sum(axis=1)
        rows.append(r)
res = pd.concat(rows, ignore_index=True)
print(f"\n행정동 × 월 × 요소 {len(res):,}행 (관측소 기준: 그 달 값 {SHARE:.0%} 이상)")

print("\n[③ 가장 가까운 관측소까지 거리 km (행정동 × 월 기준 분위수)]")
q = res.groupby("element").d1.quantile([0.5, 0.9, 0.95, 0.99, 1.0]).unstack().reindex(ELEMS).round(1)
q.columns = ["50%", "90%", "95%", "99%", "최대"]
print(q.to_string())

print("\n[③-2 가장 가까운 관측소가 이 거리보다 먼 비율]")
far = pd.DataFrame({f"> {k} km": res.groupby("element").d1.apply(lambda x: (x > k).mean()) for k in (5, 10, 20, 30)})
print(far.reindex(ELEMS).map(lambda v: f"{v:.1%}").to_string())

print("\n[③-3 2·3번째로 가까운 관측소 거리 중앙값 km]")
print(res.groupby("element")[["d1", "d2", "d3"]].median().reindex(ELEMS).round(1).to_string())

print("\n[④ 반경 안 관측소 수가 0 / 1 / 2 / 3개 이상인 비율]")
for R in RADII:
    c = res[f"n{R}"].clip(upper=3)
    t = pd.crosstab(res.element, c, normalize="index").reindex(ELEMS)
    t.columns = [f"{int(v)}개" if v < 3 else "3개+" for v in t.columns]
    print(f"  반경 {R} km")
    print(t.map(lambda v: f"{v:.1%}").to_string())

print("\n[⑤ TA 기준 가장 먼 행정동 10곳 (24개월 평균 거리)]")
ta = res[res.element == "TA"]
worst = ta.groupby(["code", "name"]).d1.mean().sort_values(ascending=False).head(10).round(1)
print(worst.to_string())

print("\n[⑤-2 가장 가까운 관측소가 구역 밖(섬·바다) 관측소인 행정동 × 월 비율]")
print(res.assign(sea=res.nearest_stn.isin(SEA)).groupby("element").sea.mean().reindex(ELEMS).map(lambda v: f"{v:.2%}").to_string())