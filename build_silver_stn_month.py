"""6-2 관측소 × 월 표: 달마다 관측소의 위치(좌표·높이)를 정해 저장한다.

입력  data/bronze/stn_raw.parquet (지점정보 25시점: 매달 1일 24개 + 2025-12-31 23시)
      data/silver/aws_availability.parquet (그 달에 기상 자료가 있었나)
출력  data/silver/stn_month.parquet / .csv   관측소 × 월 한 행

위치를 정하는 규칙
  - P1: 그 달 1일 지점정보의 좌표·높이를 그 달 전체에 씀
  - P2: 그 달 1일 지점정보에 없으면(월 중간 신설) 그 뒤 첫 지점정보를 씀 → position_source = "next"
  - P3: 다음 시점의 좌표와 0.1 km 이상 다르면 "이번 달 안에 옮겼을 수 있음" 표시 (moved_in_month)
        옮긴 날짜는 지점정보로 알 수 없으므로 그 달은 1일 위치를 쓰고 표시만 함
  - LAW_ID(법정동코드)는 원래 값 그대로 보관만 함. 기준 시점이 섞여 있어 위치 판정에는 쓰지 않음
"""
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
MOVE_KM = 0.1

stn = pd.read_parquet("data/bronze/stn_raw.parquet")
stn = pd.DataFrame({
    "station_id": stn["STN_ID"].str.strip(),
    "snap": stn["src_file"].str[8:20],                              # stn_aws_YYYYMMDDHHMI.txt
    "lon": pd.to_numeric(stn["LON"], errors="coerce"),
    "lat": pd.to_numeric(stn["LAT"], errors="coerce"),
    "height_m": pd.to_numeric(stn["HT"], errors="coerce"),
    "station_name": stn["STN_KO"].str.strip(),
    "law_id_raw": stn["LAW_ID"].str.strip(),
    "law_addr_raw": stn["LAW_ADDR"].str.strip(),
    "bronze_file": "stn/" + stn["src_file"],
}).sort_values(["station_id", "snap"]).reset_index(drop=True)
snaps = sorted(stn.snap.unique())
months = pd.period_range("2024-01", "2025-12", freq="M").strftime("%Y%m").tolist()
print(f"지점정보 {len(snaps)}개 시점, 관측소 {stn.station_id.nunique()}곳")

# 그 달에 기상 자료가 있었나
av = pd.read_parquet("data/silver/aws_availability.parquet")
has = av.groupby(["station_id", "month"]).n_rows.max().gt(0).rename("has_aws_data").reset_index()


def km(lat1, lon1, lat2, lon2):
    """두 점 사이 거리(km). 짧은 거리라 평면 근사"""
    la1, la2 = np.radians(lat1), np.radians(lat2)
    dx = np.radians(lon2 - lon1) * np.cos((la1 + la2) / 2)
    return 6371 * np.hypot(dx, la2 - la1)


rows = []
for sid, g in stn.groupby("station_id"):
    g = g.set_index("snap")
    for mo in months:
        first = mo + "010000"
        later = [s for s in g.index if s >= first]
        if not later:
            continue                                                # 그 달 이후 지점정보에 없음 (폐지)
        pos = later[0]
        source = "first_of_month" if pos == first else "next"      # P1 / P2
        nxt = [s for s in g.index if s > pos]
        month_end = (pd.Period(mo, "M") + 1).strftime("%Y%m") + "010000"   # 다음 달 1일 (2025-12는 12-31 23시 파일이 여기에 들어감)
        r = g.loc[pos]
        move = km(r.lat, r.lon, g.loc[nxt[0]].lat, g.loc[nxt[0]].lon) if nxt else np.nan
        rows.append({"station_id": sid, "month": mo, "lon": r.lon, "lat": r.lat, "height_m": r.height_m,
                     "station_name": r.station_name, "law_id_raw": r.law_id_raw, "law_addr_raw": r.law_addr_raw,
                     "position_snapshot": pos, "position_source": source,
                     "move_km_next": round(float(move), 3) if nxt else np.nan,
                     "moved_in_month": bool(nxt) and move >= MOVE_KM and nxt[0] <= month_end,   # P3
                     "bronze_file": r.bronze_file})
sm = pd.DataFrame(rows)

# 그 달 1일에도 없고 그 달 안에 새 지점정보도 없는데 다음 달 이후에만 있는 경우 = 아직 생기지 않은 관측소 → 자료가 없으면 뺌
sm = sm.merge(has, on=["station_id", "month"], how="left")
sm["has_aws_data"] = sm.has_aws_data.fillna(False).astype(bool)
not_yet = (sm.position_source == "next") & (sm.position_snapshot.str[:6] > sm.month) & ~sm.has_aws_data
sm = sm[~not_yet].reset_index(drop=True)

# 자료는 있는데 표에 없는 관측소×월이 없어야 함
miss = has[has.has_aws_data].merge(sm[["station_id", "month"]], how="left", indicator=True)
miss = miss[miss._merge == "left_only"]
assert len(miss) == 0, f"자료는 있는데 위치를 못 정한 관측소×월 {len(miss)}개: {miss.head().values.tolist()}"

cols = ["station_id", "month", "lon", "lat", "height_m", "station_name", "position_snapshot", "position_source",
        "moved_in_month", "move_km_next", "has_aws_data", "law_id_raw", "law_addr_raw", "bronze_file"]
sm = sm[cols].sort_values(["station_id", "month"]).reset_index(drop=True)
sm.to_parquet("data/silver/stn_month.parquet", index=False)
sm.to_csv("data/silver/stn_month.csv", index=False, encoding="utf-8-sig")

print(f"저장: data/silver/stn_month.parquet/.csv {len(sm):,}행, 관측소 {sm.station_id.nunique()}곳")
print(f"  기상 자료 있는 관측소×월 {int(sm.has_aws_data.sum()):,} / 지점정보만 있는 관측소×월 {int((~sm.has_aws_data).sum()):,}")
print(f"  P2 (월 중간 신설, 다음 시점 위치 사용): {int((sm.position_source == 'next').sum())}행")
print(f"  P3 (이번 달 안에 {MOVE_KM} km 이상 옮겼을 수 있음): {int(sm.moved_in_month.sum())}행")
print(sm[sm.moved_in_month][["station_id", "station_name", "month", "move_km_next", "has_aws_data"]].sort_values("move_km_next", ascending=False).to_string(index=False))