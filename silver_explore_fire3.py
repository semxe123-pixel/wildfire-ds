"""7-1 산불 탐색 3: 산림청 주소를 '발생 시점의 행정동 코드'로 바꿀 수 있는지 확인한다. (저장 없음, 출력만)

산림청 주소 모양 (탐색 1)
  locsi    시도 약칭            예: 경북
  locgungu 시군구, 접미사 없음   예: 의성, 창원, 북, 용인처인, 천안 동남
  locmenu  읍면 또는 구, 접미사 없음, 비어 있으면 시의 동 지역   예: 안계, 마산합포
  locdong  리 또는 동, 접미사 없음   예: 용기, 노곡

방법
  변환표(adm_map)에서 발생일에 유효한 줄만 놓고, 주소 이름과 법정동 이름(접미사를 뗀 것)이 맞는 줄을 찾음
  → 그 줄들의 행정동 코드가 하나면 확정, 여러 개면 '모호'(한 법정동이 행정동 여러 곳에 걸침), 없으면 '못 찾음'
  ① 읍면 지역: 시군구 + 읍면(locmenu) + 리(locdong)
  ② 동 지역: 시군구(+ 구) + 동(locdong)

입력  data/bronze/fire_raw.parquet, data/silver/adm_map.parquet
출력  화면 출력만
"""
import re
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 40)
TZ = "Asia/Seoul"
SIDO = {"서울": "서울특별시", "부산": "부산광역시", "대구": "대구광역시", "인천": "인천광역시", "광주": "광주광역시",
        "대전": "대전광역시", "울산": "울산광역시", "세종": "세종특별자치시", "경기": "경기도", "강원": "강원특별자치도",
        "충북": "충청북도", "충남": "충청남도", "전북": "전북특별자치도", "전남": "전라남도", "경북": "경상북도",
        "경남": "경상남도", "제주": "제주특별자치도"}


def norm_sido(s):
    s = str(s).strip().replace("전라북도", "전북특별자치도").replace("강원도", "강원특별자치도")
    return SIDO.get(s, s)


def sgg_key(s):
    """시군구 이름 → 비교용: 단어마다 끝의 시·군·구를 떼고 붙임 ('천안시 동남구' → '천안동남', '의성군' → '의성')"""
    s = str(s).strip()
    if not s or s == "nan":
        return ""
    return "".join(w[:-1] if len(w) >= 2 and w[-1] in "시군구" else w for w in s.split())


def name_key(s):
    """읍면동리 이름 → 비교용: 끝의 읍·면·동·리·가를 뗌 ('용기리' → '용기', '노곡동' → '노곡')"""
    s = str(s).strip()
    if not s or s == "nan":
        return ""
    return s[:-1] if len(s) >= 2 and s[-1] in "읍면동리가" else s


def hhmm(s):
    d = re.sub(r"\D", "", str(s))
    return (d + "0000")[:4] if d else ""


# ================================================================ 산림청 주소
fr = pd.read_parquet("data/bronze/fire_raw.parquet")
fr["start"] = pd.to_datetime(fr.startyear.astype(str) + fr.startmonth.astype(str).str.zfill(2) + fr.startday.astype(str).str.zfill(2)
                             + fr.starttime.map(hhmm), format="%Y%m%d%H%M", errors="coerce").dt.tz_localize(TZ)
fr["sido"] = fr.locsi.map(norm_sido)
fr["g"] = fr.locgungu.astype(str).str.replace(" ", "", regex=False).replace("nan", "")
fr["m"] = fr.locmenu.fillna("").astype(str).str.strip()
fr["d"] = fr.locdong.fillna("").astype(str).str.strip()
print("[주소 모양] 시군구에 띄어쓰기가 있는 행:", int(fr.locgungu.astype(str).str.contains(" ").sum()),
      "/ 읍면 빈 행:", int((fr.m == "").sum()), "/ 동리 빈 행:", int((fr.d == "").sum()))

# ================================================================ 변환표: 줄마다 비교용 이름
adm = pd.read_parquet("data/silver/adm_map.parquet")
emd = adm[adm.admin_level == "emd"].copy()
if "legal_level" not in emd:
    emd["legal_level"] = emd.legal_dong_code.str[8:10].map(lambda x: "ri" if x != "00" else "emd")
emd["sido_n"] = emd.sido_name.map(norm_sido)
emd["sk"] = emd.sigungu_name.map(sgg_key)
emd["vt"] = emd.valid_to.fillna(pd.Timestamp("2100-01-01", tz=TZ))
# 리의 상위 읍면 이름: 같은 법정동 코드 앞 8자리를 가진 읍면 수준 법정동 줄의 이름
lv = adm[adm.legal_dong_code.str[8:10] == "00"].drop_duplicates("legal_dong_code")
up = pd.Series(lv.legal_dong_name.values, index=lv.legal_dong_code.str[:8].values)
up = up[~up.index.duplicated()]
emd["menu_k"] = emd.legal_dong_code.str[:8].map(up).map(name_key).where(emd.legal_level == "ri", "")
emd["dong_k"] = emd.legal_dong_name.astype(str).str.replace(r"동(\d+)가$", r"\1가", regex=True).map(name_key)   # 평화동2가 -> 평화2
emd["branch"] = emd.admin_dong_name.astype(str).str.contains("출장소")


def edit1(a, b):
    """글자 수가 같고 한 글자만 다름 (예: 새도 ↔ 세도)"""
    return len(a) == len(b) and len(a) >= 2 and sum(x != y for x, y in zip(a, b)) == 1


def same(series, name):
    """주소 이름과 법정동 이름 비교: 주소 그대로, 또는 주소에서도 접미사를 뗀 것 (남면·거촌리처럼 접미사를 남긴 주소 대비)"""
    return (series == name) | (series == name_key(name))


def find(sg, m, d):
    """시군구 후보 줄(sg) 안에서 읍면/구 m, 동리 d로 찾기 → (줄, 방법)"""
    if m:
        a = sg[(sg.legal_level == "ri") & same(sg.menu_k, m) & same(sg.dong_k, d)]
        if len(a):
            return a, "읍면+리"
        b = sg[(sg.legal_level == "emd") & same(sg.dong_k, d) & sg.sk.str.endswith(m)]
        if len(b):
            return b, "구+동"
        if not d:
            c = sg[(sg.legal_level == "emd") & same(sg.dong_k, m)]
            if len(c):
                return c, "읍면만"
    else:
        a = sg[(sg.legal_level == "emd") & same(sg.dong_k, d)]
        if len(a):
            return a, "동"
    return sg.iloc[0:0], ""


def typo(sg, m, d):
    """한 글자 오타 보정: 읍면·구와 동리 중 하나만 한 글자 다른 후보. 행정동 코드가 하나로 나올 때만"""
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
    v = emd[(emd.sido_n == r.sido) & (emd.valid_from <= r.start) & (emd.vt > r.start)]
    m = r.m
    if r.sido == "세종특별자치시":
        sg = v                                                    # 세종은 시군구가 없음. 산림청은 시군구 칸에 읍면을 적음
        m = r.m or r.g
    else:
        sg = v[(v.sk == r.g) | v.sk.str.startswith(r.g)] if r.g else v.iloc[0:0]
    hits, how = find(sg, m, r.d)
    if not len(hits) and len(r.g) > 2:                            # 시군구 칸에 구를 잘못 붙인 경우: 시 이름(앞 2글자)만으로 다시
        hits, how = find(v[v.sk.str.startswith(r.g[:2])], m, r.d)
        how = how + " (시군구 앞부분만)" if how else ""
    if not len(hits):
        hits = typo(sg, m, r.d)
        how = "한 글자 오타 보정" if len(hits) else ""
    if not len(hits) and m:                                       # 시군구가 틀린 경우: 같은 시도 전체에서 읍면+리가 하나로 나오면
        h = v[(v.legal_level == "ri") & same(v.menu_k, m) & same(v.dong_k, r.d)]
        if h.admin_dong_code.nunique() == 1:
            hits, how = h, "시도 전체 (시군구 다름)"
    if hits.branch.any() and (~hits.branch).any():               # 출장소와 원래 읍면이 함께 나오면 원래 읍면
        hits = hits[~hits.branch]
    codes = sorted(set(hits.admin_dong_code))
    return pd.Series({"how": how if len(hits) else "", "n_codes": len(codes), "codes": ",".join(codes),
                      "names": ",".join(sorted(set(hits.admin_dong_name))), "branch": bool(hits.branch.any()),
                      "sgg_found": len(sg) > 0})


res = fr.join(fr.apply(lookup, axis=1))
res["result"] = res.n_codes.map(lambda n: "확정(1곳)" if n == 1 else "모호(2곳+)" if n > 1 else "못 찾음")
print("\n[결과]", res.result.value_counts().to_dict(), "/ 찾은 방법:", res.how.value_counts().to_dict())
fix = res[res.how.str.contains("보정|앞부분|시도 전체")]
print("\n[보정으로 찾은 사건]")
print(fix[["start", "locsi", "locgungu", "locmenu", "locdong", "how", "names"]].to_string(index=False))
print("  출장소 행정동이 후보에 들어간 사건:", int(res.branch.sum()))
print("  시군구부터 못 찾은 사건:", int((~res.sgg_found).sum()), "예:", res[~res.sgg_found][["locsi", "locgungu"]].drop_duplicates().head(10).values.tolist())

cols = ["start", "locsi", "locgungu", "locmenu", "locdong", "locbunji", "damagearea", "how", "names", "codes"]
print("\n[모호한 사건 전부]")
print(res[res.n_codes > 1][cols].to_string(index=False))
print("  모호한 사건의 후보 수:", res[res.n_codes > 1].n_codes.value_counts().to_dict())
print("\n[못 찾은 사건 전부]")
print(res[res.n_codes == 0][cols[:7]].to_string(index=False))
print("\n[확정 예 10]")
print(res[res.n_codes == 1][cols].head(10).to_string(index=False))
big = res[pd.to_numeric(res.damagearea, errors="coerce") >= 100]
print("\n[피해 면적 100 ha 이상]")
print(big[cols].to_string(index=False))