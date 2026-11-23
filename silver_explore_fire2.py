"""7-1 산불 탐색 2: 산림청 사건과 소방청 화재를 잇는 규칙을 정하기 위한 확인 (저장 없음, 출력만)

탐색 1에서 남은 것
  - 산림청이 일반시의 구를 붙여 쓴 이름(예: '용인처인', '창원마산합포', '포항남')이 소방청 이름과 안 맞음 → 앞부분이 같은 이름으로 맞춤
  - 24시간 안 가장 가까운 후보 중에는 하루 가까이 차이 나는 것도 있음 → 몇 분 안이면 같은 사건으로 볼지
  - 후보가 없는 96건: 소방청이 임야가 아닌 다른 화재 종류(예: 건축물에서 번진 불)로 기록했는지
  - 소방청 한 건이 산림청 두 건에 이어지는 경우가 있는지

입력  data/bronze/fire_raw.parquet, data/bronze/fire119/*.csv
출력  화면 출력만
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 30)

SIDO = {"서울": "서울특별시", "부산": "부산광역시", "대구": "대구광역시", "인천": "인천광역시", "광주": "광주광역시",
        "대전": "대전광역시", "울산": "울산광역시", "세종": "세종특별자치시", "경기": "경기도", "강원": "강원특별자치도",
        "충북": "충청북도", "충남": "충청남도", "전북": "전북특별자치도", "전남": "전라남도", "경북": "경상북도",
        "경남": "경상남도", "제주": "제주특별자치도"}
WINDOWS = [10, 30, 60, 120]                 # 비교해 볼 시각 차이(분)


def norm_sido(s):
    s = str(s).strip().replace("전라북도", "전북특별자치도").replace("강원도", "강원특별자치도")
    return SIDO.get(s, s)


def city_nf(s):
    """소방청 시군구 → 첫 단어에서 끝의 시·군·구를 뗌 ('창원시 마산합포구' → '창원')"""
    s = str(s).strip()
    if not s or s == "nan":
        return ""
    w = s.split()[0]
    return w[:-1] if len(w) >= 2 and w[-1] in "시군구" else w


def hhmm(s):
    d = re.sub(r"\D", "", str(s))
    return (d + "0000")[:4] if d else ""


# ================================================================ 읽기
fr = pd.read_parquet("data/bronze/fire_raw.parquet")
fr["start"] = pd.to_datetime(fr.startyear.astype(str) + fr.startmonth.astype(str).str.zfill(2) + fr.startday.astype(str).str.zfill(2)
                             + fr.starttime.map(hhmm), format="%Y%m%d%H%M", errors="coerce")
fr["sido"] = fr.locsi.map(norm_sido)
fr["city_raw"] = fr.locgungu.astype(str).str.strip().str.split().str[0].fillna("")
fr["area"] = pd.to_numeric(fr.damagearea, errors="coerce")

nf = pd.concat([pd.read_csv(f, encoding="utf-8-sig", dtype=str) for f in sorted(glob.glob("data/bronze/fire119/*.csv"))], ignore_index=True)
nf["rcpt"] = pd.to_datetime(nf.rcpt_dt, format="%Y%m%d%H%M%S", errors="coerce")
nf["sido"] = nf.ctpv_nm.map(norm_sido)
nf["city"] = nf.sgg_nm.map(city_nf)
nf.loc[nf.sido == "세종특별자치시", "city"] = ""
for c in ("dth_cnt", "injpsn_cnt", "prpt_dam_amt"):
    nf[c] = pd.to_numeric(nf[c], errors="coerce")

# ================================================================ ① 시군구 이름 맞추기
names = nf.groupby("sido").city.apply(set).to_dict()


def city_fs(sido, raw):
    """산림청 시군구 → 소방청 이름. 그대로 있으면 그대로, 없으면 앞부분이 같은 가장 긴 이름 ('용인처인' → '용인')"""
    if sido == "세종특별자치시":
        return ""
    have = names.get(sido, set())
    if raw in have:
        return raw
    pref = [n for n in have if n and raw.startswith(n)]
    return max(pref, key=len) if pref else raw


fr["city"] = [city_fs(s, r) for s, r in zip(fr.sido, fr.city_raw)]
chg = fr[fr.city != fr.city_raw].drop_duplicates(["sido", "city_raw"])
print("[① 산림청 시군구 이름 바꾼 것]", chg[["sido", "city_raw", "city"]].values.tolist())
left = sorted(set(zip(fr.sido, fr.city)) - {(s, c) for s, cs in names.items() for c in cs})
print("  그래도 소방청 자료(모든 화재)에 없는 이름:", left)

# ================================================================ ② 후보 찾기 (모든 화재 종류, 앞뒤 24시간)
m = fr.reset_index().merge(nf[["sido", "city", "rcpt", "wrinv_no", "fire_type_nm", "fclt_plc_mclsf_nm", "dth_cnt", "injpsn_cnt", "prpt_dam_amt"]],
                           on=["sido", "city"], how="left")
m["gap"] = (m.rcpt - m.start).dt.total_seconds() / 60
m = m[m.gap.abs() <= 24 * 60].assign(absgap=lambda x: x.gap.abs(), forest=lambda x: x.fire_type_nm.eq("임야"))
# 임야 화재를 먼저, 그 안에서 시각이 가장 가까운 것
best = m.sort_values(["index", "forest", "absgap"], ascending=[True, False, True]).drop_duplicates("index")

print("\n[② 시각 차이별로 이어지는 산림청 사건 수 (가장 가까운 후보 기준)]")
rows = []
for w in WINDOWS:
    near = m[m.absgap <= w].sort_values(["index", "forest", "absgap"], ascending=[True, False, True]).drop_duplicates("index")
    rows.append({"차이(분)": w, "이어짐": len(near), "임야로": int(near.forest.sum()), "다른 종류로": int((~near.forest).sum()),
                 "안 이어짐": len(fr) - len(near),
                 "소방 1건이 산림청 2건 이상에": int(near.wrinv_no.duplicated(keep=False).sum())})
print(pd.DataFrame(rows).to_string(index=False))

W = 60
near = m[m.absgap <= W].sort_values(["index", "forest", "absgap"], ascending=[True, False, True]).drop_duplicates("index")
print(f"\n[③ {W}분 안에서 이어진 경우]")
print("  시각 차이(분) 분포:", near.gap.describe(percentiles=[.01, .05, .5, .95, .99]).round(1).to_dict())
print("  임야 아닌 종류로 이어진 경우의 종류:", near[~near.forest].fire_type_nm.value_counts().to_dict())
print("  소방 장소(산불/들불 등):", near.fclt_plc_mclsf_nm.value_counts().to_dict())
cand = m[m.absgap <= W].groupby("index").size()
print("  후보가 2개 이상인 산림청 사건:", int((cand >= 2).sum()))
dupw = near[near.wrinv_no.duplicated(keep=False)]
if len(dupw):
    print("  소방 1건이 산림청 여러 건에 이어진 예:")
    print(dupw.sort_values("wrinv_no")[["wrinv_no", "start", "sido", "city", "area", "rcpt", "gap"]].head(10).to_string(index=False))

print(f"\n[④ {W}분 안에서 안 이어진 산림청 사건]")
no = fr.drop(near["index"])
print(f"  {len(no)}건 / 시도별", no.sido.value_counts().head(8).to_dict())
print("  피해 면적:", no.area.describe(percentiles=[.5, .9]).round(2).to_dict())
far = best[best["index"].isin(no.index)]
print("  24시간 안에는 후보가 있었던 건:", len(far), "/ 그 후보와 시각 차이(분) 예:", far.gap.round(0).head(10).tolist())
print(no[["start", "locsi", "locgungu", "locmenu", "locdong", "area", "firecause"]].head(15).to_string(index=False))

print("\n[⑤ 같은 날 같은 시군구에 산림청 사건이 여러 개인 경우 (예: 의성 2025-03-22)]")
fr["day"] = fr.start.dt.date
multi = fr[fr.duplicated(["sido", "city", "day"], keep=False)]
print(f"  {len(multi)}건 ({multi.groupby(['sido', 'city', 'day']).ngroups}묶음)")
show = near.set_index("index")[["wrinv_no", "rcpt", "gap", "dth_cnt", "injpsn_cnt", "prpt_dam_amt"]]
ex = fr[(fr.city == "의성") & (fr.day.astype(str) == "2025-03-22")].join(show)
print(ex[["start", "locmenu", "locdong", "area", "wrinv_no", "rcpt", "gap", "dth_cnt", "injpsn_cnt", "prpt_dam_amt"]].to_string())