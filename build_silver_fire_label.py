"""Silver 7-1: 산불 Label 표 만들기 → data/silver/fire_label.parquet / .csv

한 줄 = 산림청 산불 사건 1건 (738건). 학습·평가의 단위가 되는 표.

하는 일
  1) 산림청 사건 정리: 발생·진화 시각(KST), 피해 면적(ha), 원인, 주소 원문
  2) 주소 → 발생 시점에 유효한 행정동 코드 (규칙 A0~A6, 탐색 3과 같은 방법)
       A0 변환표(adm_map)에서 발생 시각에 유효한 줄만 사용. 읍면 지역은 시군구+읍면+리, 동 지역은 시군구(+구)+동
       A1 주소 쪽에 접미사가 남아 있어도(남면, 거촌리) 떼고 비교
       A2 '○동N가'는 '○N가'로 비교 (평화동2가 = 평화2가)
       A3 못 찾으면 시군구 이름의 앞 2글자(시 이름)만으로 다시
       A4 못 찾으면 한 글자 오타 보정. 행정동 코드가 하나로 나올 때만 인정
       A5 못 찾으면 같은 시도 전체에서 읍면+리가 행정동 하나로 나오면 인정 (시군구를 잘못 적은 경우)
       A8 그래도 못 찾으면 읍면을 무시하고 시군구 안에서 리 이름(그대로, 안 되면 한 글자 오타)이
          법정동 하나·행정동 하나로만 나올 때 인정 (예: 포항북 홍해 홍안 → 흥해읍 흥안리)
       A6 출장소 행정동과 원래 읍면이 함께 나오면 원래 읍면
       A9 출장소 행정동만 나오면 원래 읍면으로 바꿈: 2024-11 경계에서 그 출장소 구역의 90% 이상을 품은 행정동
          (관측소 매핑 B3 근거: 출장소 60개 모두 원래 읍·면 안에 100%. 대표점·후보 관측소는 출장소를 빼고 만들었으므로
           출장소 코드로 두면 Gold와 이어지지 않음). 그 코드가 발생 시점에 유효할 때만 바꿈
       A7 그래도 모호(행정동 2곳 이상)·못 찾음이면 후보 목록만 남김.
          data/silver/fire_geocode.parquet(7-2 지번 좌표 찾기 결과)가 있으면 그 행정동으로 채움
  3) 소방청 화재 기록과 연결 (규칙 K1~K5) → 사망·부상·재산피해
       K1 같은 시도 + 같은 시군구 (소방청 이름은 접미사를 떼고, 산림청 '용인처인' 같은 이름은 앞부분이 같은 이름으로. 세종은 시도만)
       K2 |소방 접수 시각 - 산림청 발생 시각| ≤ 60분
       K3 임야 화재를 먼저, 그 안에서 시각 차이가 가장 작은 것. 소방 1건은 산림청 1건에만
       K4 연결 안 된 사건의 사망·부상·재산피해는 0이 아니라 빈 값(NaN)
       K5 산림청에 없는 소방청 임야 화재는 넣지 않음

입력  data/bronze/fire_raw.parquet, data/bronze/fire119/*.csv, data/silver/adm_map.parquet
      data/bronze/boundary/202411/*.zip (A9: 출장소가 있을 때만 읽음)
      (있으면) data/silver/fire_geocode.parquet
출력  data/silver/fire_label.parquet, fire_label.csv
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 40)
TZ = "Asia/Seoul"
LINK_MIN = 60                                   # K2
OUT = "data/silver/fire_label"
GEOCODE = "data/silver/fire_geocode.parquet"
SIDO = {"서울": "서울특별시", "부산": "부산광역시", "대구": "대구광역시", "인천": "인천광역시", "광주": "광주광역시",
        "대전": "대전광역시", "울산": "울산광역시", "세종": "세종특별자치시", "경기": "경기도", "강원": "강원특별자치도",
        "충북": "충청북도", "충남": "충청남도", "전북": "전북특별자치도", "전남": "전라남도", "경북": "경상북도",
        "경남": "경상남도", "제주": "제주특별자치도"}


def norm_sido(s):
    s = str(s).strip().replace("전라북도", "전북특별자치도").replace("강원도", "강원특별자치도")
    return SIDO.get(s, s)


def sgg_key(s):
    """시군구 이름 → 비교용: 단어마다 끝의 시·군·구를 떼고 붙임 ('천안시 동남구' → '천안동남')"""
    s = str(s).strip()
    if not s or s == "nan":
        return ""
    return "".join(w[:-1] if len(w) >= 2 and w[-1] in "시군구" else w for w in s.split())


def name_key(s):
    """읍면동리 이름 → 비교용: 끝의 읍·면·동·리·가를 뗌 ('용기리' → '용기')"""
    s = str(s).strip()
    if not s or s == "nan":
        return ""
    return s[:-1] if len(s) >= 2 and s[-1] in "읍면동리가" else s


def hhmm(s):
    d = re.sub(r"\D", "", str(s))
    return (d + "0000")[:4] if d else ""


def to_time(y, mo, d, t):
    s = y.astype(str) + mo.astype(str).str.zfill(2) + d.astype(str).str.zfill(2) + t.map(hhmm)
    return pd.to_datetime(s, format="%Y%m%d%H%M", errors="coerce").dt.tz_localize(TZ)


# ================================================================ 1) 산림청 사건
fr = pd.read_parquet("data/bronze/fire_raw.parquet")
fr["ignition_at"] = to_time(fr.startyear, fr.startmonth, fr.startday, fr.starttime)
fr["extinguished_at"] = to_time(fr.endyear, fr.endmonth, fr.endday, fr.endtime) if "endyear" in fr else pd.NaT
fr["sido"] = fr.locsi.map(norm_sido)
fr["g"] = fr.locgungu.astype(str).str.replace(" ", "", regex=False).replace("nan", "")
fr["m"] = fr.locmenu.fillna("").astype(str).str.strip()
fr["d"] = fr.locdong.fillna("").astype(str).str.strip()
# 사건 번호: 발생 시각 → 주소 순으로 정렬해 연도별 일련번호 (같은 입력이면 늘 같은 번호)
fr = fr.sort_values(["ignition_at", "sido", "g", "m", "d", "locbunji"], kind="stable").reset_index(drop=True)
fr["fire_id"] = fr.ignition_at.dt.year.astype(str) + "-" + (fr.groupby(fr.ignition_at.dt.year).cumcount() + 1).astype(str).str.zfill(4)

# ================================================================ 2) 주소 → 행정동
adm = pd.read_parquet("data/silver/adm_map.parquet")
emd = adm[adm.admin_level == "emd"].copy()
if "legal_level" not in emd:
    emd["legal_level"] = emd.legal_dong_code.str[8:10].map(lambda x: "ri" if x != "00" else "emd")
emd["sido_n"] = emd.sido_name.map(norm_sido)
emd["sk"] = emd.sigungu_name.map(sgg_key)
emd["vt"] = emd.valid_to.fillna(pd.Timestamp("2100-01-01", tz=TZ))
lv = adm[adm.legal_dong_code.str[8:10] == "00"].drop_duplicates("legal_dong_code")
up = pd.Series(lv.legal_dong_name.values, index=lv.legal_dong_code.str[:8].values)       # 리의 상위 읍면 이름
up = up[~up.index.duplicated()]
emd["menu_k"] = emd.legal_dong_code.str[:8].map(up).map(name_key).where(emd.legal_level == "ri", "")
emd["dong_k"] = emd.legal_dong_name.astype(str).str.replace(r"동(\d+)가$", r"\1가", regex=True).map(name_key)   # A2
emd["branch"] = emd.admin_dong_name.astype(str).str.contains("출장소")


def edit1(a, b):
    return len(a) == len(b) and len(a) >= 2 and sum(x != y for x, y in zip(a, b)) == 1


def same(series, name):
    return (series == name) | (series == name_key(name))                                  # A1


def find(sg, m, d):
    if m:
        a = sg[(sg.legal_level == "ri") & same(sg.menu_k, m) & same(sg.dong_k, d)]
        if len(a):
            return a, "emd_ri"
        b = sg[(sg.legal_level == "emd") & same(sg.dong_k, d) & sg.sk.str.endswith(m)]
        if len(b):
            return b, "gu_dong"
        if not d:
            c = sg[(sg.legal_level == "emd") & same(sg.dong_k, m)]
            if len(c):
                return c, "emd_only"
    else:
        a = sg[(sg.legal_level == "emd") & same(sg.dong_k, d)]
        if len(a):
            return a, "dong"
    return sg.iloc[0:0], ""


def typo(sg, m, d):                                                                       # A4
    dk, mk = name_key(d), name_key(m)
    near_d = sg.dong_k.map(lambda x: edit1(x, d) or edit1(x, dk)).astype(bool)
    if m:
        hit = sg[(sg.legal_level == "ri") & ((same(sg.menu_k, m) & near_d)
                                             | (sg.menu_k.map(lambda x: edit1(x, m) or edit1(x, mk)).astype(bool) & same(sg.dong_k, d)))]
        if not len(hit):
            hit = sg[(sg.legal_level == "emd") & sg.sk.str.endswith(m) & near_d]
        if not len(hit):
            near_m = sg.sk.map(lambda x: any(edit1(x[i:i + len(m)], m) for i in range(len(x) - len(m) + 1))).astype(bool)
            hit = sg[(sg.legal_level == "emd") & near_m & same(sg.dong_k, d)]
    else:
        hit = sg[(sg.legal_level == "emd") & near_d]
    return hit if hit.admin_dong_code.nunique() == 1 else sg.iloc[0:0]


def lookup(r):
    v = emd[(emd.sido_n == r.sido) & (emd.valid_from <= r.ignition_at) & (emd.vt > r.ignition_at)]   # A0
    m = r.m
    if r.sido == "세종특별자치시":
        sg, m = v, (r.m or r.g)
    else:
        sg = v[(v.sk == r.g) | v.sk.str.startswith(r.g)] if r.g else v.iloc[0:0]
    hits, how = find(sg, m, r.d)
    if not len(hits) and len(r.g) > 2:                                                    # A3
        hits, how = find(v[v.sk.str.startswith(r.g[:2])], m, r.d)
        how = "city_prefix" if len(hits) else ""
    if not len(hits):
        hits = typo(sg, m, r.d)
        how = "typo1" if len(hits) else ""
    if not len(hits) and m:                                                               # A5
        h = v[(v.legal_level == "ri") & same(v.menu_k, m) & same(v.dong_k, r.d)]
        if h.admin_dong_code.nunique() == 1:
            hits, how = h, "sido_wide"
    if not len(hits) and m and r.d:                                                       # A8: 읍면을 무시하고 리 이름만
        rr = sg[sg.legal_level == "ri"]
        h = rr[same(rr.dong_k, r.d)]
        if h.legal_dong_code.nunique() != 1:
            h = rr[rr.dong_k.map(lambda x: edit1(x, r.d) or edit1(x, name_key(r.d))).astype(bool)]
        if h.legal_dong_code.nunique() == 1 and h.admin_dong_code.nunique() == 1:
            hits, how = h, "ri_only"
    if hits.branch.any() and (~hits.branch).any():                                        # A6
        hits = hits[~hits.branch]
    codes = sorted(set(hits.admin_dong_code))
    one = hits.drop_duplicates("admin_dong_code")
    return pd.Series({
        "addr_mapping_method": how if len(hits) else "",
        "admin_dong_candidates": ",".join(codes),
        "n_candidates": len(codes),
        "admin_dong_code": codes[0] if len(codes) == 1 else None,
        "admin_dong_name": one.admin_dong_name.iloc[0] if len(codes) == 1 else None,
        "sigungu_name": one.sigungu_name.iloc[0] if len(codes) == 1 else None,
        "legal_dong_code": hits.legal_dong_code.iloc[0] if hits.legal_dong_code.nunique() == 1 else None,
    })


fr = fr.join(fr.apply(lookup, axis=1))
fr["addr_mapping_status"] = np.select([fr.n_candidates == 1, fr.n_candidates > 1], ["confirmed", "ambiguous"], "not_found")
fr["addr_is_estimated"] = fr.addr_mapping_method.isin(["city_prefix", "typo1", "sido_wide", "ri_only"])
fr.loc[fr.addr_mapping_status != "confirmed", "addr_mapping_method"] = ""
fr["addr_text_status"] = fr.addr_mapping_status          # 주소 글자만으로 낸 결과 (지번 좌표로 채우기 전). 7-2가 이 열로 대상을 고름
print("[주소 → 행정동]", fr.addr_mapping_status.value_counts().to_dict(), "/ 방법:", fr.addr_mapping_method.value_counts().to_dict())

# A7: 7-2 지번 좌표 찾기 결과가 있으면 모호·못 찾음 사건을 채움 (좌표 찾기 쪽에서 받아들인 결과만 geo_admin_dong_code가 있음)
fr["geocode_lon"] = np.nan
fr["geocode_lat"] = np.nan
fr["geocode_in_candidates"] = pd.NA
if os.path.exists(GEOCODE):
    gc = pd.read_parquet(GEOCODE).set_index("fire_id")
    todo = fr.addr_mapping_status.ne("confirmed") & fr.fire_id.isin(gc.index[gc.geo_admin_dong_code.notna()])
    g = gc.loc[fr.loc[todo, "fire_id"]]
    fr.loc[todo, "admin_dong_code"] = g.geo_admin_dong_code.values
    fr.loc[todo, "admin_dong_name"] = g.geo_admin_dong_name.values
    fr.loc[todo, "sigungu_name"] = g.geo_sigungu_name.values
    fr.loc[todo, "geocode_lon"] = g.geo_lon.values
    fr.loc[todo, "geocode_lat"] = g.geo_lat.values
    fr.loc[todo, "geocode_in_candidates"] = [c in s.split(",") if s else pd.NA
                                             for c, s in zip(g.geo_admin_dong_code.values, fr.loc[todo, "admin_dong_candidates"])]
    fr.loc[todo, "addr_mapping_method"] = "geocode"
    fr.loc[todo, "addr_mapping_status"] = "confirmed"
    print(f"[지번 좌표로 채움] {int(todo.sum())}건 / 후보 목록과 다른 행정동:",
          int((fr.geocode_in_candidates == False).sum()))   # noqa: E712
else:
    print("[지번 좌표 결과 없음] fire_geocode.parquet 없음 → 모호·못 찾음은 후보만 남김 (7-2 뒤 다시 실행)")

# A9: 출장소 코드 → 그 구역을 품은 원래 읍면 (2024-11 경계에만 출장소 구역이 있음)
branch_codes = set(emd.admin_dong_code[emd.branch])
fr["branch_to_parent"] = False
isb = fr.admin_dong_code.isin(branch_codes)
if isb.any():
    import zipfile
    import geopandas as gpd
    from shapely import make_valid
    folder = "data/bronze/boundary/202411"
    parts = []
    for z in sorted(x for x in os.listdir(folder) if x.lower().endswith(".zip")):
        path = os.path.join(folder, z)
        with zipfile.ZipFile(path) as zf:
            shp = [n for n in zf.namelist() if n.endswith("TL_SCCO_GEMD.shp")]
        parts += [gpd.read_file(f"zip://{path}!{x}", encoding="cp949") for x in shp]
    gb = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs="EPSG:5179")
    gb["EMD_CD"] = gb.EMD_CD.astype(str).str.strip()
    gb["geometry"] = make_valid(gb.geometry.values)
    is_br = gb.EMD_KOR_NM.str.contains("출장소")
    br = gb[is_br & gb.EMD_CD.isin(set(fr.admin_dong_code[isb]))][["EMD_CD", "geometry"]].rename(columns={"EMD_CD": "br_cd"})
    inter = gpd.overlay(br, gb[~is_br][["EMD_CD", "geometry"]], how="intersection", keep_geom_type=True)
    inter["share"] = inter.area / inter.br_cd.map(br.set_index("br_cd").area)
    best = inter.sort_values("share", ascending=False).drop_duplicates("br_cd").set_index("br_cd")
    names_all = emd.drop_duplicates("admin_dong_code").set_index("admin_dong_code")
    for i in fr.index[isb]:
        b = fr.at[i, "admin_dong_code"]
        if b not in best.index or best.at[b, "share"] < 0.9:
            continue
        p = best.at[b, "EMD_CD"]
        t = fr.at[i, "ignition_at"]
        ok_p = ((emd.admin_dong_code == p) & (emd.valid_from <= t) & (emd.vt > t)).any()
        if ok_p:
            fr.loc[i, ["admin_dong_code", "admin_dong_name", "sigungu_name", "branch_to_parent"]] = \
                [p, names_all.at[p, "admin_dong_name"], names_all.at[p, "sigungu_name"], True]
    print(f"[A9 출장소 → 원래 읍면] 출장소 코드 {int(isb.sum())}건 중 바꿈 {int(fr.branch_to_parent.sum())}건")

# ================================================================ 3) 소방청 연결
nf = pd.concat([pd.read_csv(f, encoding="utf-8-sig", dtype=str).assign(fire119_bronze_file="fire119/" + os.path.basename(f))
                for f in sorted(glob.glob("data/bronze/fire119/*.csv"))], ignore_index=True)   # 계보: 어느 원본 CSV의 행인지
nf["rcpt"] = pd.to_datetime(nf.rcpt_dt, format="%Y%m%d%H%M%S", errors="coerce").dt.tz_localize(TZ)
nf["sido"] = nf.ctpv_nm.map(norm_sido)
w0 = nf.sgg_nm.fillna("").astype(str).str.strip().str.split().str[0].fillna("")
nf["city"] = w0.map(lambda w: w[:-1] if len(w) >= 2 and w[-1] in "시군구" else w)
nf.loc[nf.sido == "세종특별자치시", "city"] = ""
names = nf.groupby("sido").city.apply(set).to_dict()


def city_fs(sido, raw):
    if sido == "세종특별자치시":
        return ""
    have = names.get(sido, set())
    if raw in have:
        return raw
    pref = [n for n in have if n and raw.startswith(n)]
    return max(pref, key=len) if pref else raw


fr["city"] = [city_fs(s, str(r).split()[0] if str(r).strip() else "") for s, r in zip(fr.sido, fr.locgungu.astype(str))]
keep = ["sido", "city", "rcpt", "wrinv_no", "fire_type_nm", "fclt_plc_mclsf_nm", "dth_cnt", "injpsn_cnt", "prpt_dam_amt",
        "fire119_bronze_file"]
pair = fr[["fire_id", "sido", "city", "ignition_at"]].merge(nf[keep], on=["sido", "city"])   # K1
pair["gap"] = (pair.rcpt - pair.ignition_at).dt.total_seconds() / 60
pair = pair[pair.gap.abs() <= LINK_MIN].assign(absgap=lambda x: x.gap.abs(), forest=lambda x: x.fire_type_nm.eq("임야"))  # K2
pair = pair.sort_values(["forest", "absgap"], ascending=[False, True])                    # K3
used_f, used_w, pick = set(), set(), []
for i, r in zip(pair.index, pair[["fire_id", "wrinv_no"]].itertuples(index=False)):
    if r.fire_id not in used_f and r.wrinv_no not in used_w:
        used_f.add(r.fire_id)
        used_w.add(r.wrinv_no)
        pick.append(i)
lk = pair.loc[pick].set_index("fire_id")
lk = lk.rename(columns={"wrinv_no": "fire119_wrinv_no", "rcpt": "fire119_received_at", "gap": "fire119_gap_min",
                        "fire_type_nm": "fire119_fire_type", "fclt_plc_mclsf_nm": "fire119_place",
                        "dth_cnt": "deaths", "injpsn_cnt": "injuries", "prpt_dam_amt": "property_damage_kkrw"})
lk = lk[["fire119_wrinv_no", "fire119_received_at", "fire119_gap_min", "fire119_fire_type", "fire119_place",
         "deaths", "injuries", "property_damage_kkrw", "fire119_bronze_file"]]
for c in ("deaths", "injuries", "property_damage_kkrw"):
    lk[c] = pd.to_numeric(lk[c], errors="coerce")
fr = fr.join(lk, on="fire_id")                                                            # K4: 없으면 NaN
fr["fire119_linked"] = fr.fire119_wrinv_no.notna()
forest_all = int(nf.fire_type_nm.eq("임야").sum())
print(f"[소방청 연결] {int(fr.fire119_linked.sum())}/{len(fr)}건 이어짐 (임야 {int(fr.fire119_fire_type.eq('임야').sum())}) "
      f"/ 산림청에 없는 소방청 임야 화재 {forest_all - int(fr.fire119_fire_type.eq('임야').sum())}건은 넣지 않음 (K5)")

# ================================================================ 4) 정리
fr["damage_area_ha"] = pd.to_numeric(fr.damagearea, errors="coerce")
fr["duration_h"] = ((fr.extinguished_at - fr.ignition_at).dt.total_seconds() / 3600).round(3)
fr["qc_time_flag"] = np.select([fr.ignition_at.isna(), fr.extinguished_at.isna(), fr.duration_h < 0],
                               ["no_ignition_time", "no_end_time", "end_before_start"], "")
fr["year"] = fr.ignition_at.dt.year
# 설명서: 원본 파일 경로의 표준 계보 필드는 bronze_file. 기존 이름(src_file)은 이 필드로 매핑하고 원본은 고치지 않음
# 값 형식은 manifest.bronze_file과 같음 (data/bronze/ 아래 경로) → manifest와 바로 이어 볼 수 있음
fr["bronze_file"] = "fire/" + fr["src_file"].astype(str)
out = fr.rename(columns={"sido": "sido_name_full", "locsi": "sido_raw", "locgungu": "sigungu_raw", "locmenu": "emd_raw",
                         "locdong": "ri_dong_raw", "locbunji": "bunji_raw", "firecause": "fire_cause"})
cols = ["fire_id", "year", "ignition_at", "extinguished_at", "duration_h", "qc_time_flag",
        "sido_raw", "sigungu_raw", "emd_raw", "ri_dong_raw", "bunji_raw", "sido_name_full",
        "admin_dong_code", "admin_dong_name", "sigungu_name", "legal_dong_code",
        "addr_mapping_status", "addr_text_status", "addr_mapping_method", "addr_is_estimated", "branch_to_parent",
        "n_candidates", "admin_dong_candidates",
        "geocode_lon", "geocode_lat", "geocode_in_candidates",
        "damage_area_ha", "fire_cause",
        "fire119_linked", "fire119_wrinv_no", "fire119_received_at", "fire119_gap_min", "fire119_fire_type", "fire119_place",
        "deaths", "injuries", "property_damage_kkrw", "bronze_file", "fire119_bronze_file"]
out = out[[c for c in cols if c in out.columns]].rename(columns={"sido_name_full": "sido_name"})

# ================================================================ 5) 검사
assert out.fire_id.is_unique and len(out) == len(fr)
assert not out.fire119_wrinv_no.dropna().duplicated().any(), "소방 1건이 산림청 2건 이상에 이어짐"
ok = out[out.admin_dong_code.notna()].merge(emd[["admin_dong_code", "valid_from", "vt"]].drop_duplicates(),
                                            on="admin_dong_code")
ok = ok[(ok.valid_from <= ok.ignition_at) & (ok.vt > ok.ignition_at)].fire_id.nunique()
print(f"[검사] 행정동 코드가 발생 시점에 유효: {ok}/{int(out.admin_dong_code.notna().sum())}")

os.makedirs("data/silver", exist_ok=True)
out.to_parquet(OUT + ".parquet", index=False)
out.to_csv(OUT + ".csv", index=False, encoding="utf-8-sig")

print(f"\n[저장] {OUT}.parquet / .csv  {len(out)}행 × {out.shape[1]}열")
print("\n[연도별]")
print(out.groupby("year").agg(사건=("fire_id", "size"), 행정동확정=("admin_dong_code", lambda s: int(s.notna().sum())),
                              소방연결=("fire119_linked", "sum"), 면적합_ha=("damage_area_ha", "sum"),
                              사망=("deaths", "sum"), 부상=("injuries", "sum")).to_string())
print("\n[Label 값 분포]")
print("  피해 면적(ha):", out.damage_area_ha.describe(percentiles=[.5, .9, .99]).round(2).to_dict())
print("  면적 빈 값:", int(out.damage_area_ha.isna().sum()), "/ 0 ha:", int((out.damage_area_ha == 0).sum()))
print("  재산피해(천원, 연결된 사건):", out.property_damage_kkrw.describe(percentiles=[.5, .9]).round(0).to_dict())
print("  시각 이상:", out.qc_time_flag.value_counts().to_dict())
print("  진화 시간(h, 시각 이상 제외):", out.duration_h[out.qc_time_flag == ""].describe(percentiles=[.5, .9, .99]).round(2).to_dict())
print("\n[모호·못 찾음 사건 (7-2 지번 좌표 찾기 대상)]")
left = out[out.addr_mapping_status != "confirmed"]
print(f"  {len(left)}건 / 피해 면적 합 {left.damage_area_ha.sum():.2f} ha")
print(left[["fire_id", "sido_raw", "sigungu_raw", "emd_raw", "ri_dong_raw", "bunji_raw", "addr_mapping_status",
            "n_candidates", "damage_area_ha"]].to_string(index=False))
print("\n[피해 면적 100 ha 이상]")
print(out[out.damage_area_ha >= 100][["fire_id", "ignition_at", "sigungu_raw", "ri_dong_raw", "admin_dong_name",
                                      "damage_area_ha", "duration_h", "deaths", "injuries", "property_damage_kkrw"]].to_string(index=False))