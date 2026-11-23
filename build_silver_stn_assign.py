"""6-4 행정동 대표값 배정: 행정동 × 날짜 범위 × 요소마다 '가까운 순 후보 관측소'를 정한다.

입력  data/bronze/boundary/<기준월>/*.zip    행정동 경계 TL_SCCO_GEMD (6-3과 같은 방식: 5179, make_valid, 출장소 제외)
      data/silver/adm_map.parquet            법정동 → 행정동 + 유효기간 (코드를 그날 코드로 맞출 때)
      data/silver/stn_adm.parquet            관측소 × 날짜 범위 좌표, 구역 밖 여부 (6-3)
      data/silver/stn_month.parquet          관측소 높이 (6-2)
      data/silver/aws_availability.parquet   관측소 × 월 × 요소별 값이 남은 비율 (5단계)
출력  data/silver/adm_point.parquet / .csv   행정동 × 날짜 범위: 그날 코드, 이름, 대표점, 경계 출처
      data/silver/stn_assign.parquet         행정동 × 날짜 범위 × 요소 × 순위: 후보 관측소, 거리 (행이 많아 csv 없음)

배정 규칙
  D1 대표점: 행정동 구역의 무게중심. 무게중심이 구역 밖이면 구역 안의 점(representative_point)
  D2 후보 관측소: 그 달 그 요소 값이 하나라도 남은 관측소 (value_share > 0)
     → 결측이 많은 관측소는 Gold에서 값이 비는 시각에 자동으로 건너뜀
  D3 거리 제한 30 km (QC에서 '같은 날씨로 비교할 수 있는 거리'로 쓴 기준과 같음)
  D4 (Gold에서 적용) 매 시각 값이 있는 가장 가까운 후보 1개의 값을 씀. station_distance_km = 그 시각 쓴 관측소 거리
  D5 후보는 가까운 순 최대 5개. station_count = 반경 30 km 안 후보 관측소 수
  D6 구역 밖(섬·바다) 관측소도 후보에 포함 (station_outside로 표시)
  한계: 행정동 높이 자료가 없어 고도 차이는 보정하지 않음 (station_height_m을 남김)

코드 맞춤 (6-3 B4·B5·B6을 경계 구역에 적용)
  - 2025-06까지 2024-11 경계, 2025-07부터 2025-12 경계
  - 구역 코드가 그날 유효하지 않으면 법정동 다리 → 이름 다리로 그날 코드로 바꿈
  - 두 구역이 그날 같은 코드가 되면 합침 (예: 2025-12 경계의 운서1동·운서2동 → 2025년 운서동)
  - R1 영역 다리: 그래도 못 맞춘 구역이 다른 경계 파일의 한 동 안에 90% 이상 들어가면 그 동의 일부로 합침
       (예: 2025-12 경계에 미리 그려진 2026년 신설 동 → 2025년 원래 동)
       R1-2: 한 동 안에 90% 이상 들지 않으면, 다른 경계 파일 기준으로 각 동에 속한 조각(1% 이상)을 그 동에 붙임
       (예: 여러 동의 일부를 떼어 2025-10에 새로 생긴 동 → 그 전에는 떼어 온 각 동으로 되돌림)
  - R2 그날 유효한데 경계 파일에 없는 동은 다른 경계 파일의 구역을 씀 (polygon_source = other_boundary)
  - R3 시군구 짝 다리: 남은 '못 맞춘 구역'과 '구역 없는 동'이 같은 시군구에 하나씩이면 짝지음 (이름·법정동 코드가 함께 바뀐 경우)
  - R4 겹침 깎기: 다른 경계 파일 구역과 1% 넘게 겹치는 이 경계 파일 구역은 겹친 만큼 뺌 (trimmed = True)
  - 한 달 안에서 코드가 바뀌면 그 날짜로 행을 나눔 (예: 전북 2024-01-18)
"""
import os
import sys
import warnings
import zipfile
from functools import lru_cache

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import make_valid, union_all

warnings.filterwarnings("ignore", message=".*winding order.*")
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)
TZ, CRS, B = "Asia/Seoul", "EPSG:5179", "data/bronze/boundary"
BRANCH = "출장소"
RADIUS_KM = 30                                                      # D3
TOP_K = 5                                                           # D5
ELEMS = ["TA", "WD", "WS", "RN_DAY", "RN_HR1", "HM", "PA", "PS"]
VERSIONS = {
    "202411": ("MOIS_GEMD_202411", pd.Timestamp("2024-01-01", tz=TZ), pd.Timestamp("2025-07-01", tz=TZ)),
    "202512": ("MOIS_GEMD_202512", pd.Timestamp("2025-07-01", tz=TZ), pd.Timestamp("2026-01-01", tz=TZ)),
}
OTHER = {"202411": "202512", "202512": "202411"}


def version_of(month):
    return "202411" if month <= "202506" else "202512"


# ================================================================ 1. 경계 읽기
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
    g = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=CRS)
    g["EMD_CD"] = g.EMD_CD.astype(str).str.strip()
    g["geometry"] = make_valid(g.geometry.values)
    g = g[~g.EMD_KOR_NM.str.contains(BRANCH)].reset_index(drop=True)
    return g[["EMD_CD", "EMD_KOR_NM", "boundary_file", "geometry"]]


GEMD = {v: read_gemd(v) for v in VERSIONS}

# ================================================================ 2. 변환표: 그날 코드로 맞추기 (6-3 B5와 같음)
adm = pd.read_parquet("data/silver/adm_map.parquet")
emd = adm[(adm.admin_level == "emd") & ~adm.admin_dong_name.astype(str).str.contains(BRANCH)].copy()
emd["sido_norm"] = emd.sido_name.astype(str).str.replace("전라북도", "전북특별자치도")
emd["vt"] = emd.valid_to.fillna(pd.Timestamp("2100-01-01", tz=TZ))
LEGAL_OF = emd.groupby("admin_dong_code").legal_dong_code.apply(set).to_dict()
NAME_OF = emd.drop_duplicates("admin_dong_code").set_index("admin_dong_code")[["sido_norm", "sigungu_name", "admin_dong_name"]]
CHANGE_DATES = sorted(set(emd.valid_from.dropna()) | set(emd.valid_to.dropna()))


@lru_cache(maxsize=None)
def valid_on(day):
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
def resolve(code, day):
    """경계 구역 코드 → 그날 코드와 방법. 안 되면 (None, 'unresolved')"""
    if code in valid_codes(day):
        return code, "boundary"
    li = legal_index(day)
    cand = set().union(*[li.get(l, set()) for l in LEGAL_OF.get(code, set())]) if code in LEGAL_OF else set()
    if len(cand) == 1:
        return next(iter(cand)), "legal_bridge"
    if code in NAME_OF.index:
        cand = name_index(day).get(tuple(NAME_OF.loc[code]), set())
        if len(cand) == 1:
            return next(iter(cand)), "name_bridge"
    return None, "unresolved"


# ================================================================ 3. 그날의 행정동 구역과 대표점 (D1)
def tag(ver, day):
    g = GEMD[ver]
    r = [resolve(c, day) for c in g.EMD_CD]
    return g.assign(code=[x[0] for x in r], how=[x[1] for x in r])


def dissolve(t, ver, source):
    """같은 그날 코드가 된 구역을 합침"""
    rows = []
    for code, grp in t.groupby("code", sort=False):
        rows.append({
            "admin_dong_code": code,
            "geometry": grp.geometry.iloc[0] if len(grp) == 1 else union_all(grp.geometry.values),
            "n_polygons": len(grp),
            "polygon_codes": ",".join(sorted(grp.EMD_CD)),
            "code_mapping": ",".join(sorted(set(grp.how))),
            "ver": ver, "polygon_source": source,
            "boundary_file": ",".join(sorted(set(grp.boundary_file))),
        })
    return rows


OVERLAP_MIN = 0.9          # R1: 못 맞춘 구역이 다른 경계 파일의 한 동 안에 90% 이상 들어가면 그 동의 일부로 봄
EVENTS = []                # 확인 출력용: 다리를 놓거나 깎은 기록


@lru_cache(maxsize=None)
def dong_points(ver, day):
    t = tag(ver, day)
    o_all = tag(OTHER[ver], day)
    o_ok = o_all[o_all.code.notna()].reset_index(drop=True)

    # R1 영역 다리: 그날 코드로 못 맞춘 구역(예: 2026년에 생길 동) → 다른 경계 파일에서 그 땅이 속한 그날 동
    add, drop = [], []
    for i in t.index[t.code.isna()]:
        geom = t.geometry[i]
        cand = o_ok.sindex.query(geom, predicate="intersects")
        if len(cand) == 0:
            continue
        inter = o_ok.geometry.iloc[cand].intersection(geom)
        share = inter.area.values / geom.area
        j = int(np.argmax(share))
        if share[j] >= OVERLAP_MIN:
            t.loc[i, ["code", "how"]] = [o_ok.code.iloc[cand[j]], "overlap_bridge"]
            EVENTS.append(("overlap_bridge", ver, day, t.EMD_CD[i], o_ok.code.iloc[cand[j]], round(float(share[j]), 3)))
            continue
        # R1-2 영역 나누기: 여러 동에 걸친 구역이면, 다른 경계 파일 기준으로 각 동에 속한 조각을 그 동에 붙임
        keep = share >= 0.01                                              # 경계선 차이로 생긴 1% 미만 조각은 버림
        if share[keep].sum() >= OVERLAP_MIN:
            for k in np.flatnonzero(keep):
                piece = t.loc[[i]].copy()
                piece["geometry"] = [inter.iloc[k]]
                piece["code"], piece["how"] = o_ok.code.iloc[cand[k]], "overlap_split"
                add.append(piece)
            drop.append(i)
            EVENTS.append(("overlap_split", ver, day, t.EMD_CD[i],
                           ",".join(o_ok.code.iloc[cand[np.flatnonzero(keep)]]),
                           ",".join(f"{v:.3f}" for v in share[keep])))
    if drop:
        t = gpd.GeoDataFrame(pd.concat([t.drop(index=drop)] + add, ignore_index=True), geometry="geometry", crs=CRS)

    rows = dissolve(t[t.code.notna()], ver, "same")
    have = {r["admin_dong_code"] for r in rows}

    # R2 다른 경계 파일: 그날 유효한데 이 경계 파일에 없는 동
    missing = valid_codes(day) - have
    if missing:
        rows += dissolve(o_ok[o_ok.code.isin(missing)], OTHER[ver], "other_boundary")
        have = {r["admin_dong_code"] for r in rows}

    # R3 시군구 짝 다리: 그래도 남은 '못 맞춘 구역'과 '구역 없는 동'이 같은 시군구에 하나씩이면 짝지음
    #    (이름과 법정동 코드가 함께 바뀐 경우. 예: 면 이름 변경, 면 → 읍 승격)
    left = t[t.code.isna()]
    still = sorted(valid_codes(day) - have)
    for sgg in {c[:5] for c in still}:
        miss_s = [c for c in still if c[:5] == sgg]
        poly_s = left[left.EMD_CD.str[:5] == sgg]
        if len(miss_s) == 1 and len(poly_s) == 1:
            k = poly_s.index[0]
            t.loc[k, ["code", "how"]] = [miss_s[0], "sigungu_pair"]
            rows += dissolve(t.loc[[k]], ver, "same")
            EVENTS.append(("sigungu_pair", ver, day, t.EMD_CD[k], miss_s[0], None))
    unresolved = tuple(sorted(t.loc[t.code.isna(), "EMD_CD"]))

    d = gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS)
    d["trimmed"] = False
    # R4 겹침 깎기: 다른 경계 파일 구역과 겹치는 이 경계 파일 구역은 겹친 만큼 뺌
    #    (예: 분리 전 모양인 부모 동 구역에서 새로 생긴 동 구역을 뺌)
    oth = d.index[d.polygon_source == "other_boundary"]
    same = d[d.polygon_source == "same"]
    for i in oth:
        g = d.geometry[i]
        for j in same.index[same.sindex.query(g, predicate="intersects")]:
            ov = d.geometry[j].intersection(g).area
            if ov > 0.01 * d.geometry[j].area:                       # 그리는 방식 차이로 생긴 가장자리 겹침은 무시
                d.loc[j, "geometry"] = d.geometry[j].difference(g)
                d.loc[j, "trimmed"] = True
                EVENTS.append(("trim", ver, day, d.admin_dong_code[j], d.admin_dong_code[i], round(ov / 1e6, 3)))

    still = tuple(sorted(valid_codes(day) - set(d.admin_dong_code)))
    cen = d.geometry.centroid
    inside = cen.within(d.geometry).values
    pt = gpd.GeoSeries(np.where(inside, cen.values, d.geometry.representative_point().values), crs=CRS, index=d.index)
    ll = pt.to_crs("EPSG:4326")
    d["x"], d["y"] = pt.x.values, pt.y.values
    d["admin_centroid_lon"], d["admin_centroid_lat"] = ll.x.round(6).values, ll.y.round(6).values
    d["centroid_method"] = np.where(inside, "centroid", "inside_point")
    d["area_km2"] = (d.geometry.area / 1e6).round(3)
    return pd.DataFrame(d.drop(columns="geometry")), unresolved, still


# ================================================================ 4. 관측소 × 월 위치와 후보 자격 (D2·D6)
sa = pd.read_parquet("data/silver/stn_adm.parquet")
st = sa.drop_duplicates(["station_id", "month"])[["station_id", "month", "lon", "lat", "spatial_mapping_method"]]
hm = pd.read_parquet("data/silver/stn_month.parquet", columns=["station_id", "month", "height_m"])
st = st.merge(hm, on=["station_id", "month"], how="left")
xy = gpd.GeoSeries(gpd.points_from_xy(st.lon, st.lat), crs="EPSG:4326").to_crs(CRS)
st["x"], st["y"] = xy.x.values, xy.y.values
st["station_outside"] = st.spatial_mapping_method.ne("within")
av = pd.read_parquet("data/silver/aws_availability.parquet")
AVAIL = av[av.value_share > 0].groupby(["month", "element"]).station_id.apply(set).to_dict()   # D2

# ================================================================ 5. 행정동 × 날짜 범위 × 요소 → 후보 (D3·D5)
pt_rows, as_rows, log = [], [], []
for month in sorted(st.month.unique()):
    ver = version_of(month)
    m0 = pd.Timestamp(month[:4] + "-" + month[4:] + "-01", tz=TZ)
    m1 = m0 + pd.offsets.MonthBegin(1)
    cuts = [m0] + [c for c in CHANGE_DATES if m0 < c < m1] + [m1]
    s_m = st[st.month == month]
    for a, b in zip(cuts[:-1], cuts[1:]):
        d, unresolved, still = dong_points(ver, a)
        log.append({"month": month, "valid_from": a, "valid_codes": len(valid_codes(a)), "dongs": len(d),
                    "merged": int((d.n_polygons > 1).sum()), "other_boundary": int((d.polygon_source == "other_boundary").sum()),
                    "unresolved_polygons": unresolved, "no_polygon": still})
        p = d.assign(valid_from=a, valid_to=b, month=month)
        p["admin_boundary_version"] = p.ver.map(lambda v: VERSIONS[v][0])
        p["admin_boundary_valid_from"] = p.ver.map(lambda v: VERSIONS[v][1])
        p["admin_boundary_valid_to"] = p.ver.map(lambda v: VERSIONS[v][2])
        pt_rows.append(p)
        for el in ELEMS:
            s = s_m[s_m.station_id.isin(AVAIL.get((month, el), set()))]
            if s.empty:
                continue
            dist = np.hypot(d.x.values[:, None] - s.x.values[None, :], d.y.values[:, None] - s.y.values[None, :]) / 1000
            cnt = (dist <= RADIUS_KM).sum(axis=1)                       # station_count
            k = min(TOP_K, dist.shape[1])
            idx = np.argpartition(dist, k - 1, axis=1)[:, :k]
            dd = np.take_along_axis(dist, idx, axis=1)
            o = np.argsort(dd, axis=1)
            idx, dd = np.take_along_axis(idx, o, axis=1), np.take_along_axis(dd, o, axis=1)
            f = pd.DataFrame({
                "admin_dong_code": np.repeat(d.admin_dong_code.to_numpy(dtype=object), k),
                "valid_from": a, "valid_to": b, "month": month, "element": el,
                "rank": np.tile(np.arange(1, k + 1, dtype="int8"), len(d)),
                "station_id": s.station_id.to_numpy(dtype=object)[idx].ravel(),
                "distance_km": dd.ravel().round(3),
                "station_height_m": s.height_m.to_numpy(dtype=float, na_value=np.nan)[idx].ravel(),
                "station_outside": s.station_outside.to_numpy(dtype=bool)[idx].ravel(),
                "station_count": np.repeat(cnt, k),
            })
            as_rows.append(f[f.distance_km <= RADIUS_KM])

pts = pd.concat(pt_rows, ignore_index=True)
nm = NAME_OF.reindex(pts.admin_dong_code)
pts["admin_dong_name"], pts["sigungu_name"], pts["sido_name"] = nm.admin_dong_name.values, nm.sigungu_name.values, nm.sido_norm.values
pcols = ["admin_dong_code", "valid_from", "valid_to", "month", "admin_dong_name", "sigungu_name", "sido_name",
         "admin_centroid_lon", "admin_centroid_lat", "centroid_method", "area_km2",
         "code_mapping", "n_polygons", "polygon_codes", "polygon_source", "trimmed",
         "admin_boundary_version", "admin_boundary_valid_from", "admin_boundary_valid_to", "boundary_file"]
pts = pts[pcols].sort_values(["admin_dong_code", "valid_from"]).reset_index(drop=True)
asg = pd.concat(as_rows, ignore_index=True).sort_values(["admin_dong_code", "valid_from", "element", "rank"]).reset_index(drop=True)
asg = asg.merge(pts[["admin_dong_code", "valid_from", "admin_boundary_version"]], on=["admin_dong_code", "valid_from"], how="left")



def collapse(df):
    """코드가 바뀐 날짜로 나눈 범위 중, 이 행정동에는 아무것도 달라지지 않은 범위를 다시 이어 붙임"""
    keys = [c for c in df.columns if c not in ("valid_from", "valid_to")]
    out = (df.groupby(keys, dropna=False, sort=False)
             .agg(valid_from=("valid_from", "min"), valid_to=("valid_to", "max"), n=("valid_from", "size"))
             .reset_index())
    return out


n_seg = len(pts)
pts = collapse(pts)
asg = collapse(asg)
pts = pts[pcols].sort_values(["admin_dong_code", "valid_from"]).reset_index(drop=True)
acols = ["admin_dong_code", "valid_from", "valid_to", "month", "element", "rank", "station_id", "distance_km",
         "station_height_m", "station_outside", "station_count", "admin_boundary_version"]
asg = asg[acols].sort_values(["admin_dong_code", "valid_from", "element", "rank"]).reset_index(drop=True)
same = pts.admin_dong_code.eq(pts.admin_dong_code.shift())
gap = same & pts.valid_from.ne(pts.valid_to.shift())
print(f"코드가 바뀐 날짜로 나눈 범위 {n_seg:,} → 달라진 것이 없는 범위를 이어 붙여 {len(pts):,}")
print(f"같은 행정동에서 앞 행의 끝과 다음 행의 시작이 다른 곳: {int(gap.sum())} (행정동이 새로 생기거나 없어진 경우)")

os.makedirs("data/silver", exist_ok=True)
pts.to_parquet("data/silver/adm_point.parquet", index=False)
pts.to_csv("data/silver/adm_point.csv", index=False, encoding="utf-8-sig")
asg.to_parquet("data/silver/stn_assign.parquet", index=False)
print(f"저장: data/silver/adm_point.parquet/.csv {len(pts):,}행 (행정동 × 날짜 범위)")
print(f"저장: data/silver/stn_assign.parquet {len(asg):,}행 (행정동 × 날짜 범위 × 요소 × 순위)")

# ================================================================ 6. 확인 출력
lg = pd.DataFrame(log)
print("\n[행정동 구역] 그날 유효한 행정동 수 vs 대표점을 만든 행정동 수 (날짜 범위별)")
print(lg[["month", "valid_from", "valid_codes", "dongs", "merged", "other_boundary"]]
      .assign(valid_from=lg.valid_from.dt.strftime("%m-%d"),
              no_polygon=lg.no_polygon.map(len), unresolved_polygons=lg.unresolved_polygons.map(len)).to_string(index=False))
nop = sorted({c for t in lg.no_polygon for c in t})
print(f"\n  구역을 못 찾은 행정동 {len(nop)}곳:", [(c, *NAME_OF.loc[c].tolist()) if c in NAME_OF.index else c for c in nop][:15])
unr = sorted({c for t in lg.unresolved_polygons for c in t})
print(f"  그날 코드로 못 맞춘 경계 구역 {len(unr)}개:", unr[:15])
mg = pts[pts.n_polygons > 1].drop_duplicates(["admin_dong_code", "polygon_codes"])
print("  합친 구역 예:", mg.head(8)[["month", "admin_dong_code", "admin_dong_name", "polygon_codes"]].values.tolist())
ob = pts[pts.polygon_source == "other_boundary"].drop_duplicates("admin_dong_code")
print("  다른 경계 파일 구역을 쓴 행정동:", ob.head(8)[["month", "admin_dong_code", "admin_dong_name", "admin_boundary_version"]].values.tolist())
ev = pd.DataFrame(EVENTS, columns=["rule", "ver", "day", "from_code", "to_code", "value"]).drop_duplicates(["rule", "from_code", "to_code"])
for rule, label in [("overlap_bridge", "R1 영역 다리 (구역 코드 → 그날 동, 겹침 비율)"),
                    ("overlap_split", "R1-2 영역 나누기 (구역 코드 → 그날 동들, 조각 비율)"), ("sigungu_pair", "R3 시군구 짝 다리 (구역 코드 → 그날 동)"),
                    ("trim", "R4 겹침 깎기 (깎인 동 ← 다른 경계 구역 동, 깎인 넓이 km²)")]:
    x = ev[ev.rule == rule]
    nm_ = lambda c: "/".join(NAME_OF.admin_dong_name.get(x, "?") for x in str(c).split(","))
    print(f"  {label} {len(x)}건:", [(r.day.strftime("%Y-%m-%d"), r.from_code, nm_(r.from_code), "→", r.to_code, nm_(r.to_code), r.value)
                                    for r in x.head(10).itertuples()])
print("  코드 맞춤:", pts.code_mapping.value_counts().to_dict())
print("  대표점 방법:", pts.drop_duplicates("admin_dong_code").centroid_method.value_counts().to_dict())

print("\n[후보 관측소] 행정동 × 날짜 범위 × 요소 기준")
key = ["admin_dong_code", "valid_from", "element"]
g = asg.groupby(key).agg(n=("rank", "size"), d1=("distance_km", "first"), sc=("station_count", "first"),
                         out1=("station_outside", "first")).reset_index()
full = pts[["admin_dong_code", "valid_from"]].merge(pd.DataFrame({"element": ELEMS}), how="cross")
miss = full.merge(g[key], on=key, how="left", indicator=True)
print("  후보가 하나도 없는 행정동 × 범위 × 요소:", miss[miss._merge == "left_only"].groupby("element").size().reindex(ELEMS, fill_value=0).to_dict())
summ = g.groupby("element").agg(
    후보5개=("n", lambda x: f"{(x == TOP_K).mean():.1%}"),
    후보1개=("n", lambda x: f"{(x == 1).mean():.2%}"),
    d1_50=("d1", lambda x: round(x.quantile(0.5), 1)),
    d1_99=("d1", lambda x: round(x.quantile(0.99), 1)),
    d1_max=("d1", "max"),
    station_count_50=("sc", "median"),
    station_count_min=("sc", "min"),
    구역밖이_1순위=("out1", lambda x: f"{x.mean():.2%}"),
).reindex(ELEMS)
print(summ.to_string())
print("\n  [예] 4580033000 (전북 부안 행안면) 2024-01 TA:")
print(asg[(asg.admin_dong_code.isin(["4580033000", "5280033000"])) & (asg.month == "202401") & (asg.element == "TA")]
      [["admin_dong_code", "valid_from", "valid_to", "rank", "station_id", "distance_km", "station_count"]].to_string(index=False))