"""6-3 관측소 → 소재지 행정동: 관측소 × 날짜 범위마다 관측소가 있는 행정동을 정한다.

입력  data/silver/stn_month.parquet          관측소 × 월 위치 (6-2)
      data/silver/adm_map.parquet            법정동 → 행정동 + 유효기간 (2단계 변환표)
      data/bronze/boundary/<기준월>/*.zip    행안부 "구역의 도형" 행정동 경계 TL_SCCO_GEMD (시도 17개 × 2시점)
출력  data/silver/stn_adm.parquet / .csv     관측소 × 날짜 범위 한 행 (valid_from 이상, valid_to 미만, 한국 시간)

판정 규칙
  B1 좌표계 EPSG:5179 (경계 파일에 좌표계 정보가 없어 데이터로 확인: 시군구 이름 일치 96.7%, 다른 후보 0%)
  B2 잘못 그려진 경계는 make_valid로 고쳐 씀 (고친 뒤 넓이 비율 1.0000)
  B3 출장소 구역은 빼고 판정 (출장소 60개 모두 원래 읍·면 안에 100%)
  B4 2025-06까지 2024-11 경계, 2025-07부터 2025-12 경계
  B5 경계에서 나온 코드가 그날 유효하지 않으면 그날 코드로 바꿈
       ① 법정동 다리: 그 코드에 속한 법정동이 그날 속한 행정동이 하나면 그것 (예: 운서1동 → 운서동)
       ② 이름 다리: 같은 시도·시군구·행정동 이름의 그날 코드가 하나면 그것 (예: 전북 52… → 45…, 법정동 코드도 함께 바뀐 경우)
       ③ 다른 경계 파일의 결과가 그날 유효하면 그것  ④ 모두 안 되면 경계 코드를 두고 "unresolved"
  B6 한 달 안에서 코드가 바뀌면 그 날짜로 행을 나눔
  B7 어느 경계에도 들어가지 않으면 가장 가까운 행정동 + 거리
       단, 지점정보 주소(시도·시군구·읍면동 이름)가 그날 유효한 다른 행정동 하나로 풀리면 주소를 따름
       (섬·바다 관측소: 가장 가까운 육지 구역은 추정일 뿐이고, 주소는 기상청이 등록한 소재지. 예: 955 용유동 → 옹진군 자월면)
       주소가 2026년 이름(영종구 등)이라 풀리지 않으면 가장 가까운 구역을 그대로 씀
  B8 달 중간에 옮긴 관측소는 그 달 1일 위치로 판정 (moved_in_month 표시 유지)
  B9 경계 버전·유효기간·매핑 방법 열을 남김
  B10 is_estimated: 경계 안이고 코드도 그대로면 False, 그 밖(경계 밖, 주소, 코드 다리)은 True
      mapping_confidence: 관측소 둘레 반경 100 m 원 중 판정된 행정동 구역 안에 드는 넓이 비율 (0~1)
"""
import os
import sys
import warnings
import zipfile
from functools import lru_cache
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import make_valid

warnings.filterwarnings("ignore", message=".*winding order.*")
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)
TZ, CRS, B = "Asia/Seoul", "EPSG:5179", "data/bronze/boundary"
BRANCH = "출장소"
BUFFER_M = 100                                                      # B10: 6-2에서 0.1 km 미만 좌표 변화는 정밀도 보정이었음
VERSIONS = {                                                        # B4
    "202411": ("MOIS_GEMD_202411", pd.Timestamp("2024-01-01", tz=TZ), pd.Timestamp("2025-07-01", tz=TZ)),
    "202512": ("MOIS_GEMD_202512", pd.Timestamp("2025-07-01", tz=TZ), pd.Timestamp("2026-01-01", tz=TZ)),
}


def version_of(month):
    return "202411" if month <= "202506" else "202512"


# ================================================================ 1. 경계 읽기 (B1·B2·B3)
def read_gemd(ver):
    parts = []
    folder = os.path.join(B, ver)
    for z in sorted(f for f in os.listdir(folder) if f.lower().endswith(".zip")):
        path = os.path.join(folder, z)
        with zipfile.ZipFile(path) as zf:
            shp = [n for n in zf.namelist() if n.endswith("TL_SCCO_GEMD.shp")]
        for s in shp:
            g = gpd.read_file(f"zip://{path}!{s}", encoding="cp949")
            g["boundary_file"] = f"boundary/{ver}/{z}"
            parts.append(g)
    g = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=CRS)   # B1
    g["EMD_CD"] = g.EMD_CD.astype(str).str.strip()
    n_bad = int((~g.geometry.is_valid).sum())
    g["geometry"] = make_valid(g.geometry.values)                                             # B2
    branch = g.EMD_KOR_NM.str.contains(BRANCH)
    print(f"[{ver}] 행정동 구역 {len(g):,} / 고친 경계 {n_bad} / 뺀 출장소 {int(branch.sum())}")
    return g[~branch].reset_index(drop=True)                                                  # B3


GEMD = {v: read_gemd(v) for v in VERSIONS}

# ================================================================ 2. 변환표: 코드 유효기간, 법정동·이름 다리
adm = pd.read_parquet("data/silver/adm_map.parquet")
emd = adm[(adm.admin_level == "emd") & ~adm.admin_dong_name.astype(str).str.contains(BRANCH)].copy()
emd["sido_norm"] = emd.sido_name.astype(str).str.replace("전라북도", "전북특별자치도")
FAR = pd.Timestamp("2100-01-01", tz=TZ)
emd["vt"] = emd.valid_to.fillna(FAR)
LEGAL_OF = emd.groupby("admin_dong_code").legal_dong_code.apply(set).to_dict()                # 코드에 속한 적 있는 법정동
NAME_OF = emd.drop_duplicates("admin_dong_code").set_index("admin_dong_code")[["sido_norm", "sigungu_name", "admin_dong_name"]]
CHANGE_DATES = sorted(set(emd.valid_from.dropna()) | set(emd.valid_to.dropna()))              # B6: 코드가 바뀔 수 있는 날짜


@lru_cache(maxsize=None)
def valid_on(day):
    """그날 유효한 관계 행 (출장소 제외)"""
    return emd[(emd.valid_from <= day) & (emd.vt > day)]


@lru_cache(maxsize=None)
def valid_codes(day):
    return frozenset(valid_on(day).admin_dong_code)


@lru_cache(maxsize=None)
def legal_index(day):
    return valid_on(day).groupby("legal_dong_code").admin_dong_code.apply(set).to_dict()


@lru_cache(maxsize=None)
def name_index(day):
    v = valid_on(day).drop_duplicates("admin_dong_code")
    return v.groupby(["sido_norm", "sigungu_name", "admin_dong_name"]).admin_dong_code.apply(set).to_dict()


@lru_cache(maxsize=None)
def resolve(code, other_code, day):
    """B5: 그날 쓸 코드와 방법"""
    if code in valid_codes(day):
        return code, "boundary"
    li = legal_index(day)
    cand = set().union(*[li.get(l, set()) for l in LEGAL_OF.get(code, set())]) if code in LEGAL_OF else set()
    if len(cand) == 1:
        return next(iter(cand)), "legal_bridge"                                               # ①
    if code in NAME_OF.index:
        k = tuple(NAME_OF.loc[code])
        cand = name_index(day).get(k, set())
        if len(cand) == 1:
            return next(iter(cand)), "name_bridge"                                            # ②
    if other_code is not None and other_code in valid_codes(day):
        return other_code, "other_boundary"                                                   # ③
    return code, "unresolved"                                                                 # ④


@lru_cache(maxsize=None)
def addr_table(day):
    v = valid_on(day)
    return pd.DataFrame({
        "sido": v.sido_norm.astype(str).values,
        "sgg": v.sigungu_name.fillna("").astype(str).str.replace(" ", "", regex=False).values,   # 세종은 시군구 없음
        "dong": v.admin_dong_name.astype(str).values,
        "legal": v.legal_dong_name.astype(str).values,
        "code": v.admin_dong_code.values,
    })


@lru_cache(maxsize=None)
def resolve_addr(addr, day):
    """B7 보완: 주소의 시도·시군구·읍면동(또는 법정동) 이름이 그날 유효한 행정동 하나로 풀리면 그 코드, 아니면 None"""
    if not isinstance(addr, str) or not addr.strip():
        return None
    toks = addr.replace("(산지)", " ").replace("전라북도", "전북특별자치도").split()
    flat = "".join(toks)
    t = addr_table(day)
    ok = ((t.sido == toks[0])
          & t.sgg.map(lambda x: x == "" or x in flat)
          & (t.dong.isin(toks) | t.legal.isin(toks)))
    cand = set(t.code[ok])
    return next(iter(cand)) if len(cand) == 1 else None


# ================================================================ 3. 관측소 좌표 → 경계 구역 (B7·B8·B10)
sm = pd.read_parquet("data/silver/stn_month.parquet")
sm["version"] = sm.month.map(version_of)
pts = sm[["lon", "lat"]].drop_duplicates().reset_index(drop=True)                             # 같은 좌표는 한 번만 판정
geo = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts.lon, pts.lat), crs="EPSG:4326").to_crs(CRS)
HIT = {}
for ver, g in GEMD.items():
    cols = g[["EMD_CD", "boundary_file", "geometry"]]
    j = gpd.sjoin(geo, cols, predicate="within", how="left")
    n_hit = j.groupby(level=0).EMD_CD.count()
    if (n_hit > 1).any():
        print(f"[{ver}] 출장소를 빼고도 구역 두 개에 걸친 좌표 {int((n_hit > 1).sum())}개 → 첫 구역 사용")
    j = j[~j.index.duplicated()]
    inside = j.EMD_CD.notna()
    near = gpd.sjoin_nearest(geo[~inside.values], cols, distance_col="dist")                  # B7
    near = near[~near.index.duplicated()]
    res = pd.DataFrame({"code": j.EMD_CD, "file": j.boundary_file, "method": np.where(inside, "within", "nearest"), "dist": 0.0})
    res.loc[near.index, ["code", "file", "dist"]] = near[["EMD_CD", "boundary_file", "dist"]].values
    # B10 mapping_confidence: 반경 100 m 원 중 판정된 구역 안에 드는 비율
    poly = g.set_index("EMD_CD").geometry
    circles = geo.geometry.buffer(BUFFER_M, quad_segs=32)
    inter = circles.intersection(gpd.GeoSeries(poly.reindex(res.code).values, index=res.index, crs=CRS))
    res["conf"] = (inter.area / circles.area).round(4)
    HIT[ver] = res
POLY = {ver: dict(zip(g.EMD_CD, g.geometry)) for ver, g in GEMD.items()}


def geom_stats(pt, ver, code):
    """주소로 정한 구역까지의 거리(m)와 반경 100 m 비율. 그 경계 파일에 코드가 없으면 NaN"""
    poly = POLY[ver].get(code)
    if poly is None:
        return np.nan, np.nan
    p = geo.geometry.iloc[pt]
    c = p.buffer(BUFFER_M, quad_segs=32)
    return round(float(p.distance(poly)), 1), round(float(c.intersection(poly).area / c.area), 4)
key = {(x, y): i for i, (x, y) in enumerate(zip(pts.lon, pts.lat))}
sm["pt"] = [key[(x, y)] for x, y in zip(sm.lon, sm.lat)]

# ================================================================ 4. 관측소 × 날짜 범위 (B5·B6)
rows = []
for r in sm.itertuples():
    h = HIT[r.version].loc[r.pt]
    other = HIT["202512" if r.version == "202411" else "202411"].loc[r.pt, "code"]
    m0 = pd.Timestamp(r.month[:4] + "-" + r.month[4:] + "-01", tz=TZ)
    m1 = m0 + pd.offsets.MonthBegin(1)
    cuts = [m0] + [d for d in CHANGE_DATES if m0 < d < m1] + [m1]
    segs = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        code, how = resolve(h.code, other, a)
        meth = h.method
        if meth == "nearest":                                                                 # B7 보완: 경계 밖이면 주소 확인
            ac = resolve_addr(r.law_addr_raw, a)
            if ac is not None and ac != code:
                code, how, meth = ac, "address", "address"
        if segs and segs[-1][2:] == [code, how, meth]:
            segs[-1][1] = b                                                                   # 결과가 같으면 이어 붙임
        else:
            segs.append([a, b, code, how, meth])
    ver_name, bv_from, bv_to = VERSIONS[r.version]
    for a, b, code, how, meth in segs:
        if meth == "address":
            dist, conf = geom_stats(r.pt, r.version, code)
        else:
            dist, conf = round(float(h.dist), 1), h.conf
        rows.append({
            "station_id": r.station_id, "valid_from": a, "valid_to": b, "month": r.month,
            "station_admin_dong_code": code, "boundary_admin_dong_code": h.code,
            "spatial_mapping_method": meth, "code_mapping": how,
            "is_estimated": not (meth == "within" and how == "boundary"),
            "mapping_confidence": conf, "distance_to_boundary_m": dist,
            "admin_boundary_version": ver_name, "admin_boundary_valid_from": bv_from, "admin_boundary_valid_to": bv_to,
            "lon": r.lon, "lat": r.lat, "moved_in_month": r.moved_in_month, "has_aws_data": r.has_aws_data,
            "position_snapshot": r.position_snapshot, "stn_bronze_file": r.bronze_file, "boundary_file": h.file,
        })
out = pd.DataFrame(rows)
nm = NAME_OF.reindex(out.station_admin_dong_code)
out["admin_dong_name"] = nm.admin_dong_name.values
out["sigungu_name"] = nm.sigungu_name.values
out["sido_name"] = nm.sido_norm.values

cols = ["station_id", "valid_from", "valid_to", "station_admin_dong_code", "admin_dong_name", "sigungu_name", "sido_name",
        "spatial_mapping_method", "code_mapping", "is_estimated", "mapping_confidence", "distance_to_boundary_m",
        "boundary_admin_dong_code", "admin_boundary_version", "admin_boundary_valid_from", "admin_boundary_valid_to",
        "month", "lon", "lat", "moved_in_month", "has_aws_data", "position_snapshot", "stn_bronze_file", "boundary_file"]
out = out[cols].sort_values(["station_id", "valid_from"]).reset_index(drop=True)

# 확인: 같은 관측소의 날짜 범위가 겹치거나 비지 않아야 함
same_stn = out.station_id.eq(out.station_id.shift())
broken = same_stn & out.valid_from.ne(out.valid_to.shift())
print(f"\n같은 관측소에서 앞 행의 끝과 다음 행의 시작이 다른 곳: {int(broken.sum())}곳 (지점정보에 없던 달이 있는 관측소면 끊길 수 있음)")

out.to_parquet("data/silver/stn_adm.parquet", index=False)
out.to_csv("data/silver/stn_adm.csv", index=False, encoding="utf-8-sig")
print(f"저장: data/silver/stn_adm.parquet/.csv {len(out):,}행, 관측소 {out.station_id.nunique()}곳 (관측소×월 {len(sm):,} → 날짜로 나눈 뒤 {len(out):,})")

# ================================================================ 5. 확인 출력
d = out[out.has_aws_data]
print("\n[기상 자료가 있는 행 기준]")
print("  공간 판정:", d.spatial_mapping_method.value_counts().to_dict())
print("  코드 맞춤:", d.code_mapping.value_counts().to_dict())
print("  is_estimated:", d.is_estimated.value_counts().to_dict())
print("  mapping_confidence 분포:", d.mapping_confidence.describe(percentiles=[0.01, 0.05, 0.5])[["min", "1%", "5%", "50%"]].round(3).to_dict(),
      f"/ 1.0인 행 {(d.mapping_confidence >= 0.9999).mean():.1%}")
for how in ("legal_bridge", "name_bridge", "other_boundary", "address", "unresolved"):
    x = d[d.code_mapping == how].drop_duplicates(["station_id", "station_admin_dong_code"])
    if len(x):
        print(f"  [{how}] 관측소 {x.station_id.nunique()}곳 예:",
              x.head(6)[["station_id", "month", "boundary_admin_dong_code", "station_admin_dong_code", "admin_dong_name"]].values.tolist())
split = out.groupby(["station_id", "month"]).size()
print(f"  한 달 안에서 나뉜 관측소×월: {int((split > 1).sum())}")
print("  경계 밖 관측소 (B7):", d[d.spatial_mapping_method.isin(["nearest", "address"])].drop_duplicates("station_id")[["station_id", "spatial_mapping_method", "admin_dong_name", "distance_to_boundary_m"]].values.tolist())

# 검증: 판정된 시군구 이름이 지점정보 주소에 들어 있나 (방법별)
sm_addr = sm.set_index(["station_id", "month"]).law_addr_raw
addr = sm_addr.reindex(pd.MultiIndex.from_frame(d[["station_id", "month"]])).str.replace(" ", "", regex=False).values
name = d.sigungu_name.where(d.sigungu_name.notna(), d.sido_name)                             # 세종은 시군구가 없어 시도 이름으로 확인
ok = np.array([isinstance(a, str) and isinstance(s, str) and s.replace(" ", "") in a for a, s in zip(addr, name)])
print("\n[검증] 판정된 시군구 이름(세종은 시도)이 지점정보 주소에 들어 있는 비율 (인천 2026년 새 구 이름 때문에 100%가 될 수 없음)")
print(pd.Series(ok).groupby([d.spatial_mapping_method.values, d.code_mapping.values]).agg(["size", "mean"]).round(3).to_string())