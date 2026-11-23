"""Silver 7-2: 주소로 행정동을 하나로 못 정한 산불 사건을 지번 좌표로 정한다 → data/silver/fire_geocode.parquet

대상  fire_label에서 addr_text_status(주소 글자만으로 낸 결과)가 ambiguous(행정동 후보 2곳 이상) 또는 not_found인 사건

방법
  G1 지번 주소 만들기
       모호: 법정동은 하나로 정해져 있음 → 변환표의 정식 이름으로 '시도 시군구 법정동 지번' (예: 부산광역시 사하구 감천동 산98)
       못 찾음: 산림청 주소 원문 그대로 '시도 시군구 읍면 리 지번' (오타가 있으면 대개 실패)
       지번 '산98'이 실패하면 '산 98'로 한 번 더
  G2 브이월드 지오코더 API (type=PARCEL, 지번주소 검색)로 좌표(경위도) 받기
       인증키는 .env의 VWORLD_KEY. 받은 응답은 data/bronze/geocode/fire_geocode_raw.jsonl에 저장(키는 저장 안 함)
       → 다시 실행하면 저장된 응답을 쓰고 API를 다시 부르지 않음
  G3 좌표 → 행정동 경계(관측소 매핑과 같은 규칙)
       B1 EPSG:5179, B2 make_valid, B3 출장소 제외, B4 2025-06까지 2024-11판 · 2025-07부터 2025-12판
       B5 경계 코드가 발생일에 유효하지 않으면 ① 법정동으로 찾기 ② 이름으로 찾기 ③ 다른 판 결과
       어느 구역에도 안 들어가면 가장 가까운 구역 + 거리
  G4 받아들이는 조건
       ① 지오코더가 찾은 법정동 코드(refined level4LC)가 우리가 찾던 법정동과 같음 (모호한 사건, 법정동을 아는 경우)
       ② 좌표로 정한 행정동이 주소로 찾은 후보 목록 안에 있음 (모호한 사건)
       ③ 못 찾음 사건은 지오코더가 돌려준 시군구 이름이 산림청 시군구와 같음
       조건을 못 지키면 geo_admin_dong_code를 비우고 이유를 geo_reject_reason에 남김

입력  data/silver/fire_label.parquet, data/silver/adm_map.parquet, data/bronze/boundary/<기준월>/*.zip, .env
출력  data/silver/fire_geocode.parquet / .csv, data/bronze/geocode/fire_geocode_raw.jsonl
다음  build_silver_fire_label.py를 다시 실행하면 이 결과가 fire_label에 들어감
"""
import json
import os
import sys
import time
import warnings
import zipfile
from functools import lru_cache

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely import make_valid

warnings.filterwarnings("ignore", message=".*winding order.*")
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 45)
TZ, CRS, B = "Asia/Seoul", "EPSG:5179", "data/bronze/boundary"
URL = "https://api.vworld.kr/req/address"
RAW = "data/bronze/geocode/fire_geocode_raw.jsonl"
OUT = "data/silver/fire_geocode"
BRANCH = "출장소"
FAR = pd.Timestamp("2100-01-01", tz=TZ)


def read_env_key(name="VWORLD_KEY"):
    """.env에서 키 한 줄 읽기 (python-dotenv 없이). 화면에 키를 찍지 않음"""
    if os.environ.get(name):
        return os.environ[name]
    if os.path.exists(".env"):
        for line in open(".env", encoding="utf-8"):
            k, _, v = line.strip().partition("=")
            if k.strip() == name:
                return v.strip().strip('"').strip("'")
    return None


# ================================================================ 1. 대상 사건과 주소 (G1)
lab = pd.read_parquet("data/silver/fire_label.parquet")
# addr_text_status: 주소 글자만으로 낸 결과. 이미 지번 좌표로 채운 사건도 다시 포함해야 다시 실행해도 결과가 줄지 않음
todo = lab[lab.addr_text_status.isin(["ambiguous", "not_found"])].copy()
adm = pd.read_parquet("data/silver/adm_map.parquet")
emd = adm[(adm.admin_level == "emd") & ~adm.admin_dong_name.astype(str).str.contains(BRANCH)].copy()
emd["sido_norm"] = emd.sido_name.astype(str).str.replace("전라북도", "전북특별자치도")
emd["vt"] = emd.valid_to.fillna(FAR)
legal_name = adm.drop_duplicates("legal_dong_code").set_index("legal_dong_code")
up = adm[adm.legal_dong_code.str[8:10] == "00"].drop_duplicates("legal_dong_code")
up = pd.Series(up.legal_dong_name.values, index=up.legal_dong_code.str[:8].values)
up = up[~up.index.duplicated()]


def build_queries(r):
    bunji = str(r.bunji_raw).strip()
    variants = [bunji]
    if bunji.startswith("산") and not bunji.startswith("산 "):
        variants.append("산 " + bunji[1:])
    if isinstance(r.legal_dong_code, str) and r.legal_dong_code in legal_name.index:
        L = legal_name.loc[r.legal_dong_code]
        sgg = str(L.sigungu_name) if str(L.sigungu_name) not in ("", "nan", "None") else ""
        if r.legal_dong_code[8:10] != "00":                             # 리: 상위 읍면 이름을 붙임
            head = f"{L.sido_name} {sgg} {up.get(r.legal_dong_code[:8], '')} {L.legal_dong_name}"
        else:
            head = f"{L.sido_name} {sgg} {L.legal_dong_name}"
        how = "legal_name"
    else:
        head = f"{r.sido_name} {r.sigungu_raw} {r.emd_raw or ''} {r.ri_dong_raw}"
        how = "raw_address"
    head = " ".join(head.split())
    return [f"{head} {v}" for v in variants], how


todo[["queries", "query_source"]] = todo.apply(lambda r: pd.Series(build_queries(r)), axis=1)
print(f"[대상] {len(todo)}건 / 주소 만든 방법:", todo.query_source.value_counts().to_dict())

# ================================================================ 2. 지오코더 (G2) — 저장된 응답이 있으면 재사용
os.makedirs(os.path.dirname(RAW), exist_ok=True)
cache = {}
if os.path.exists(RAW):
    for line in open(RAW, encoding="utf-8"):
        rec = json.loads(line)
        cache[rec["query"]] = rec["response"]
need = sorted({q for qs in todo.queries for q in qs} - set(cache))
key = read_env_key()
if need and not key:
    sys.exit("VWORLD_KEY가 .env에 없음. '.env'에 VWORLD_KEY=발급키 한 줄을 넣고 다시 실행")


def call(q):
    p = {"service": "address", "request": "getcoord", "version": "2.0", "crs": "epsg:4326", "address": q,
         "refine": "true", "simple": "false", "format": "json", "type": "parcel", "key": key}
    for attempt in range(3):
        try:
            r = requests.get(URL, params=p, timeout=15)
            r.raise_for_status()
            return r.json().get("response", {})
        except Exception as e:                                          # 일시 오류는 두 번 더
            err = type(e).__name__
            time.sleep(1 + attempt)
    return {"status": "ERROR", "error": {"text": err}}


asked = 0
with open(RAW, "a", encoding="utf-8") as f:
    for q in need:
        # 첫 지번 형태가 이미 성공한 사건의 두 번째 형태는 부르지 않음
        sibling_ok = any(qs[0] in cache and cache[qs[0]].get("status") == "OK" and q in qs[1:] for qs in todo.queries)
        if sibling_ok:
            continue
        resp = call(q)
        cache[q] = resp
        asked += 1
        f.write(json.dumps({"query": q, "asked_at": pd.Timestamp.now(tz=TZ).isoformat(), "response": resp},
                           ensure_ascii=False) + "\n")
        time.sleep(0.2)
print(f"[지오코더] 새로 부름 {asked}건 / 저장된 응답 재사용 {len(cache) - asked}건")


def pick(qs):
    for q in qs:
        resp = cache.get(q, {})
        if resp.get("status") == "OK":
            pt = resp.get("result", {}).get("point", {})
            st = resp.get("refined", {}).get("structure", {})
            return pd.Series({"geo_query": q, "geo_status": "OK", "geo_lon": float(pt.get("x")), "geo_lat": float(pt.get("y")),
                              "geo_refined": resp.get("refined", {}).get("text"),
                              "geo_legal_code": st.get("level4LC") or None,
                              "geo_vworld_admin_code": st.get("level4AC") or None,
                              "geo_sigungu_refined": st.get("level2") or ""})
    last = cache.get(qs[-1], {})
    return pd.Series({"geo_query": qs[0], "geo_status": last.get("status", "NO_RESPONSE"), "geo_lon": np.nan, "geo_lat": np.nan,
                      "geo_refined": None, "geo_legal_code": None, "geo_vworld_admin_code": None,
                      "geo_sigungu_refined": ""})


todo = todo.join(todo.queries.apply(pick))
print("  결과:", todo.geo_status.value_counts().to_dict())

# ================================================================ 3. 좌표 → 행정동 경계 (G3)
VERSIONS = {"202411": (pd.Timestamp("2024-01-01", tz=TZ), pd.Timestamp("2025-07-01", tz=TZ)),
            "202512": (pd.Timestamp("2025-07-01", tz=TZ), pd.Timestamp("2026-01-01", tz=TZ))}


def read_gemd(ver):
    parts, folder = [], os.path.join(B, ver)
    for z in sorted(x for x in os.listdir(folder) if x.lower().endswith(".zip")):
        path = os.path.join(folder, z)
        with zipfile.ZipFile(path) as zf:
            shp = [n for n in zf.namelist() if n.endswith("TL_SCCO_GEMD.shp")]
        for s in shp:
            parts.append(gpd.read_file(f"zip://{path}!{s}", encoding="cp949"))
    g = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=CRS)        # B1
    g["EMD_CD"] = g.EMD_CD.astype(str).str.strip()
    g["geometry"] = make_valid(g.geometry.values)                                                  # B2
    return g[~g.EMD_KOR_NM.str.contains(BRANCH)].reset_index(drop=True)[["EMD_CD", "geometry"]]   # B3


@lru_cache(maxsize=None)
def valid_on(day):
    return emd[(emd.valid_from <= day) & (emd.vt > day)]


LEGAL_OF = emd.groupby("admin_dong_code").legal_dong_code.apply(set).to_dict()
NAME_OF = emd.drop_duplicates("admin_dong_code").set_index("admin_dong_code")[["sido_norm", "sigungu_name", "admin_dong_name"]]


def resolve(code, other, day):
    """B5: 경계 코드 → 발생일에 유효한 코드"""
    v = valid_on(day)
    codes = set(v.admin_dong_code)
    if code in codes:
        return code, "boundary"
    li = v.groupby("legal_dong_code").admin_dong_code.apply(set).to_dict()
    cand = set().union(*[li.get(x, set()) for x in LEGAL_OF.get(code, set())]) if code in LEGAL_OF else set()
    if len(cand) == 1:
        return next(iter(cand)), "legal_bridge"
    if code in NAME_OF.index:
        k = tuple(NAME_OF.loc[code])
        vv = v.drop_duplicates("admin_dong_code")
        cand = set(vv.admin_dong_code[(vv.sido_norm == k[0]) & (vv.sigungu_name == k[1]) & (vv.admin_dong_name == k[2])])
        if len(cand) == 1:
            return next(iter(cand)), "name_bridge"
    if other is not None and other in codes:
        return other, "other_boundary"
    return code, "unresolved"


ok = todo[todo.geo_status == "OK"].copy()
for c in ("geo_boundary_code", "geo_spatial_method", "geo_code_mapping", "geo_admin_dong_code"):
    todo[c] = None
todo["geo_distance_m"] = np.nan
if len(ok):
    pts = gpd.GeoDataFrame(ok[["fire_id"]], geometry=gpd.points_from_xy(ok.geo_lon, ok.geo_lat), crs="EPSG:4326").to_crs(CRS)
    hit = {}
    for ver in VERSIONS:
        g = read_gemd(ver)
        j = gpd.sjoin(pts, g, predicate="within", how="left")
        j = j[~j.index.duplicated()]
        res = pd.DataFrame({"code": j.EMD_CD, "method": np.where(j.EMD_CD.notna(), "within", "nearest"), "dist": 0.0})
        out_pts = pts[j.EMD_CD.isna().values]
        if len(out_pts):
            near = gpd.sjoin_nearest(out_pts, g, distance_col="dist")
            near = near[~near.index.duplicated()]
            res.loc[near.index, ["code", "dist"]] = near[["EMD_CD", "dist"]].values
        hit[ver] = res
    for i, r in ok.iterrows():
        ver = "202411" if r.ignition_at < VERSIONS["202512"][0] else "202512"                    # B4
        other = "202512" if ver == "202411" else "202411"
        h = hit[ver].loc[i]
        code, how = resolve(h.code, hit[other].loc[i, "code"], r.ignition_at)
        todo.loc[i, ["geo_boundary_code", "geo_spatial_method", "geo_code_mapping", "geo_admin_dong_code"]] = [h.code, h.method, how, code]
        todo.loc[i, "geo_distance_m"] = round(float(h.dist), 1)

# ================================================================ 4. 받아들이기 (G4)
def reject(r):
    if r.geo_status != "OK":
        return "geocode_" + str(r.geo_status).lower()
    if r.geo_code_mapping == "unresolved":
        return "code_unresolved"
    if (r.query_source == "legal_name" and isinstance(r.geo_legal_code, str) and len(r.geo_legal_code) == 10
            and r.geo_legal_code != r.legal_dong_code):
        return "legal_dong_differs"                                                                # ①
    if r.addr_text_status == "ambiguous" and r.geo_admin_dong_code not in str(r.admin_dong_candidates).split(","):
        return "not_in_candidates"                                                                 # ②
    if r.addr_text_status == "not_found":
        a = "".join(str(r.sigungu_raw).split())
        b = "".join(w[:-1] if len(w) >= 2 and w[-1] in "시군구" else w for w in str(r.geo_sigungu_refined).split())
        if not (a and b and (b.startswith(a) or a.startswith(b))):
            return "sigungu_differs"                                                               # ③
    return ""


todo["geo_reject_reason"] = todo.apply(reject, axis=1)
acc = todo.geo_reject_reason == ""
names = emd.drop_duplicates("admin_dong_code").set_index("admin_dong_code")
todo["geo_admin_dong_name"] = todo.geo_admin_dong_code.map(names.admin_dong_name)
todo["geo_sigungu_name"] = todo.geo_admin_dong_code.map(names.sigungu_name)
for c in ("geo_admin_dong_code", "geo_admin_dong_name", "geo_sigungu_name"):
    todo.loc[~acc, c] = None

cols = ["fire_id", "addr_text_status", "admin_dong_candidates", "legal_dong_code", "query_source", "geo_query", "geo_status",
        "geo_refined", "geo_legal_code", "geo_vworld_admin_code", "geo_lon", "geo_lat", "geo_boundary_code", "geo_spatial_method", "geo_distance_m",
        "geo_code_mapping", "geo_admin_dong_code", "geo_admin_dong_name", "geo_sigungu_name", "geo_reject_reason"]
out = todo[cols].reset_index(drop=True)
out.to_parquet(OUT + ".parquet", index=False)
out.to_csv(OUT + ".csv", index=False, encoding="utf-8-sig")

print(f"\n[저장] {OUT}.parquet / .csv  {len(out)}행")
print(f"[받아들임] {int(acc.sum())}/{len(out)}건 / 못 받아들인 이유:", out.geo_reject_reason[out.geo_reject_reason != ""].value_counts().to_dict())
print("  경계 판정:", out.geo_spatial_method.value_counts().to_dict(), "/ 코드 바꿈:", out.geo_code_mapping.value_counts().to_dict())
show = todo[["fire_id", "sigungu_raw", "ri_dong_raw", "bunji_raw", "addr_text_status", "geo_status", "geo_refined",
             "geo_admin_dong_name", "geo_reject_reason"]]
print("\n[사건별 결과]")
print(show.to_string(index=False))