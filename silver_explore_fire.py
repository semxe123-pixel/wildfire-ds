"""7-1 산불 탐색: 산림청 산불 자료와 소방청 화재 자료를 살펴보고, 같은 사건끼리 얼마나 이어지는지 본다. (저장 없음, 출력만)

입력  data/bronze/fire_raw.parquet            산림청 산불발생통계 (대국민포털) 738행
      data/bronze/fire119/*.csv              소방청 전국 화재 현황 (받은 그대로)
출력  화면 출력만

보는 것
  ① 산림청: 열, 빈 값, 연도별 건수, 시도 이름 모양, 시각 모양, 피해 면적 분포, 원인, 같은 사건이 두 번 있는지
  ② 소방청: 임야 화재 건수 (산불·들불), 시도별 건수, 피해 열 분포
  ③ 맞대기: 산림청 사건마다 같은 시도·시군구에서 앞뒤 24시간 안에 접수된 소방청 임야 화재가 몇 건인지,
     가장 가까운 것과 시각이 얼마나 차이 나는지 → 7-2에서 연결 규칙을 정하는 근거
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 30)

# 시도 약칭 → 정식 이름 (소방청 자료의 시도 이름에 맞춤)
SIDO = {"서울": "서울특별시", "부산": "부산광역시", "대구": "대구광역시", "인천": "인천광역시", "광주": "광주광역시",
        "대전": "대전광역시", "울산": "울산광역시", "세종": "세종특별자치시", "경기": "경기도", "강원": "강원특별자치도",
        "충북": "충청북도", "충남": "충청남도", "전북": "전북특별자치도", "전남": "전라남도", "경북": "경상북도",
        "경남": "경상남도", "제주": "제주특별자치도"}


def norm_sido(s):
    s = str(s).strip()
    s = s.replace("전라북도", "전북특별자치도").replace("강원도", "강원특별자치도")
    return SIDO.get(s, s)


def city(s):
    """소방청 시군구 이름을 산림청 모양으로 맞춤: 첫 단어만, 끝의 시·군·구를 뗌
    산림청은 '창원', '의성', '북', '양구'처럼 접미사 없이 쓰고, 소방청은 '창원시 마산합포구', '의성군', '북구', '양구군'처럼 씀"""
    s = str(s).strip()
    if not s or s == "nan":
        return ""
    w = s.split()[0]
    return w[:-1] if len(w) >= 2 and w[-1] in "시군구" else w


def hhmm(s):
    """'13:20', '1320', '13' 같은 모양 → '1320'"""
    d = re.sub(r"\D", "", str(s))
    return (d + "0000")[:4] if d else ""


# ================================================================ ① 산림청
fr = pd.read_parquet("data/bronze/fire_raw.parquet")
fr.columns = [c.lower() for c in fr.columns]
print(f"[① 산림청] {len(fr)}행, 열 {list(fr.columns)}")
print("  빈 값 비율:", fr.replace("", np.nan).isna().mean().round(3)[lambda x: x > 0].to_dict())
print(fr.head(3).T.to_string())
need = ["startyear", "startmonth", "startday", "starttime", "locsi", "locgungu", "damagearea"]
miss = [c for c in need if c not in fr.columns]
if miss:
    sys.exit(f"\n산림청 자료에 예상한 열이 없음: {miss} → 위 열 목록을 붙여 주면 코드를 맞춤")

fr["start"] = pd.to_datetime(fr.startyear.astype(str).str.zfill(4) + fr.startmonth.astype(str).str.zfill(2)
                             + fr.startday.astype(str).str.zfill(2) + fr.starttime.map(hhmm),
                             format="%Y%m%d%H%M", errors="coerce")
if "endyear" in fr.columns:
    fr["end"] = pd.to_datetime(fr.endyear.astype(str).str.zfill(4) + fr.endmonth.astype(str).str.zfill(2)
                               + fr.endday.astype(str).str.zfill(2) + fr.endtime.map(hhmm),
                               format="%Y%m%d%H%M", errors="coerce")
fr["sido"] = fr.locsi.map(norm_sido)
fr["city"] = fr.locgungu.astype(str).str.strip().str.split().str[0].fillna("")   # 산림청은 이미 접미사가 없음 (예: 양구군 → '양구'라서 다시 떼면 안 됨)
fr.loc[fr.sido == "세종특별자치시", "city"] = ""          # 세종은 시군구가 없음 (소방청은 빈 값)
fr["area"] = pd.to_numeric(fr.damagearea, errors="coerce")
print("\n  발생 시각을 읽지 못한 행:", int(fr.start.isna().sum()), "/ 예:", fr.loc[fr.start.isna(), ["startyear", "startmonth", "startday", "starttime"]].head(3).values.tolist())
print("  연도별:", fr.start.dt.year.value_counts().sort_index().to_dict())
print("  시도 원래 값:", fr.locsi.value_counts().to_dict())
print("  시도를 정식 이름으로 못 바꾼 값:", sorted(set(fr.sido) - set(SIDO.values())))
print("  starttime 모양 예:", fr.starttime.astype(str).drop_duplicates().head(8).tolist())
print("  피해 면적(damagearea) 분포:", fr.area.describe(percentiles=[.5, .9, .99]).round(2).to_dict(), "/ 0인 행", int((fr.area == 0).sum()))
print("  피해 면적 큰 순 5:")
print(fr.sort_values("area", ascending=False).head(5)[["start", "locsi", "locgungu"] + [c for c in ("locmenu", "locdong") if c in fr] + ["area"]].to_string(index=False))
if "firecause" in fr.columns:
    print("  원인:", fr.firecause.value_counts().head(12).to_dict())
if "end" in fr.columns:
    dur = (fr.end - fr.start).dt.total_seconds() / 3600
    print("  진화까지 걸린 시간(시간):", dur.describe(percentiles=[.5, .9, .99]).round(1).to_dict(), "/ 음수", int((dur < 0).sum()))
addr = [c for c in ("locsi", "locgungu", "locmenu", "locdong", "locbunji") if c in fr]
dup = fr.duplicated(["start"] + addr, keep=False)
print(f"  발생 시각과 주소가 모두 같은 행: {int(dup.sum())}")

# ================================================================ ② 소방청
files = sorted(glob.glob("data/bronze/fire119/*.csv"))
nf = pd.concat([pd.read_csv(f, encoding="utf-8-sig", dtype=str).assign(src_file=os.path.basename(f)) for f in files], ignore_index=True)
print(f"\n[② 소방청] 파일 {len(files)}개, {len(nf):,}행")
fo = nf[nf.fire_type_nm == "임야"].copy()
fo["rcpt"] = pd.to_datetime(fo.rcpt_dt, format="%Y%m%d%H%M%S", errors="coerce")
fo["sido"] = fo.ctpv_nm.map(norm_sido)
fo["city"] = fo.sgg_nm.map(city)
fo.loc[fo.sido == "세종특별자치시", "city"] = ""
for c in ("dth_cnt", "injpsn_cnt", "prpt_dam_amt"):
    fo[c] = pd.to_numeric(fo[c], errors="coerce")
print(f"  임야 화재 {len(fo):,}건 / 연도별", fo.rcpt.dt.year.value_counts().sort_index().to_dict(),
      "/ 산불·들불", fo.fclt_plc_mclsf_nm.value_counts().to_dict())
print("  사망 있는 건", int((fo.dth_cnt > 0).sum()), "/ 부상 있는 건", int((fo.injpsn_cnt > 0).sum()),
      "/ 재산피해(천 원) 중앙값", fo.prpt_dam_amt.median(), "/ 최대", fo.prpt_dam_amt.max())
print("  시도를 정식 이름으로 못 바꾼 값:", sorted(set(fo.sido) - set(SIDO.values())))
only = sorted(set(zip(fr.sido, fr.city)) - set(zip(fo.sido, fo.city)))
print(f"  산림청의 시도·시군구 중 소방청 임야 화재에 한 번도 없는 것 {len(only)}개 (이름이 안 맞을 수 있음):", only[:20])

# ================================================================ ③ 맞대기
print("\n[③ 산림청 사건마다 같은 시도·시군구(첫 단어), 앞뒤 24시간 안에 접수된 소방청 임야 화재]")
m = fr.reset_index().merge(fo[["sido", "city", "rcpt", "fclt_plc_mclsf_nm", "fclt_plc_sclsf_nm", "dth_cnt", "injpsn_cnt", "prpt_dam_amt", "wrinv_no"]],
                           on=["sido", "city"], how="left")
m["gap_min"] = (m.rcpt - m.start).dt.total_seconds() / 60
m.loc[m.gap_min.abs() > 24 * 60, ["wrinv_no", "rcpt", "gap_min"]] = np.nan   # 24시간 밖은 후보 아님 (자정 넘김도 잡도록 날짜가 아닌 시간으로 비교)
cnt = m.groupby("index").wrinv_no.count()
print("  후보 수 0 / 1 / 2 / 3개 이상:", cnt.clip(upper=3).value_counts().sort_index().to_dict(), f"(산림청 {len(fr)}건)")
best = m.dropna(subset=["gap_min"]).assign(absgap=lambda x: x.gap_min.abs()).sort_values("absgap").drop_duplicates("index")
print("  가장 가까운 후보와의 시각 차이(분, 소방 접수 − 산림청 발생):",
      best.gap_min.describe(percentiles=[.1, .25, .5, .75, .9]).round(0).to_dict())
for lim in (30, 60, 120, 360):
    print(f"    차이 {lim}분 이내: {int((best.absgap <= lim).sum())}건")
print("  가장 가까운 후보의 장소:", best.fclt_plc_mclsf_nm.value_counts().to_dict())
print("  예 (시각 차이 작은 순 5, 큰 순 5):")
cols = ["start", "sido", "city", "area", "rcpt", "gap_min", "fclt_plc_sclsf_nm", "dth_cnt", "injpsn_cnt", "prpt_dam_amt"]
print(best.sort_values("absgap")[cols].head(5).to_string(index=False))
print(best.sort_values("absgap")[cols].tail(5).to_string(index=False))
no = fr.loc[cnt[cnt == 0].index]
print(f"  후보가 없는 산림청 사건 {len(no)}건, 시도별:", no.sido.value_counts().head(10).to_dict())
print("    예:", no[["start", "locsi", "locgungu", "area"]].head(5).values.tolist())
big = fr[fr.area >= 100].reset_index().merge(best[["index", "rcpt", "gap_min", "dth_cnt", "injpsn_cnt", "prpt_dam_amt"]], on="index", how="left")
print(f"  피해 면적 100 이상 산림청 사건 {len(big)}건의 연결:")
print(big[["start", "locsi", "locgungu", "area", "rcpt", "gap_min", "dth_cnt", "injpsn_cnt", "prpt_dam_amt"]].to_string(index=False))