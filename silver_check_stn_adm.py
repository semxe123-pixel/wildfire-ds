"""6-3 검증 보충: 판정한 시군구 이름이 지점정보 주소에 없는 행을 관측소별로 출력 (저장 없음)

입력: data/silver/stn_adm.parquet, data/silver/stn_month.parquet (law_addr_raw)
출력: 화면 출력만
"""
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 300)
pd.set_option("display.max_colwidth", 45)

adm = pd.read_parquet("data/silver/stn_adm.parquet")
sm = pd.read_parquet("data/silver/stn_month.parquet",
                     columns=["station_id", "month", "law_addr_raw"])

# 기상 자료가 있는 행만, 같은 관측소×월의 주소를 붙임
df = adm[adm["has_aws_data"].eq(True)].merge(sm, on=["station_id", "month"], how="left")


def name_in_addr(r):
    """띄어쓰기를 지우고, 판정한 시군구 이름이 주소 안에 있는지 확인"""
    addr = str(r.law_addr_raw).replace(" ", "")
    sgg = str(r.sigungu_name).replace(" ", "")
    return sgg in addr


df["match"] = df.apply(name_in_addr, axis=1)
bad = df[~df["match"]]

print(f"전체 {len(df):,}행 중 불일치 {len(bad):,}행 / 관측소 {bad.station_id.nunique()}곳\n")

# 관측소별로 묶어서 출력
g = (bad.groupby(["station_id", "sido_name", "sigungu_name", "admin_dong_name", "law_addr_raw"],
                 dropna=False)
        .agg(rows=("month", "size"), first=("month", "min"), last=("month", "max"),
             method=("spatial_mapping_method", "first"), code=("code_mapping", "first"))
        .reset_index()
        .sort_values(["sido_name", "sigungu_name", "station_id"]))
print(g.to_string(index=False))

print("\n[시도별 불일치 관측소 수]")
print(bad.groupby("sido_name", dropna=False).station_id.nunique().sort_values(ascending=False).to_string())