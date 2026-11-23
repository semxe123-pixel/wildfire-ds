"""6-1 관측소 탐색: 관측소 매핑을 설계하기 위한 사실 확인. 저장하지 않고 출력만 한다.

① 달마다 지점정보에 있는 관측소 수와 기상 자료가 있는 관측소 수, 서로 안 맞는 경우
② 관측소가 옮겨졌나: 지점정보 25개 시점 사이의 좌표·높이·법정동코드(LAW_ID)·이름 변화
③ 법정동코드(LAW_ID)만으로 행정동을 정할 수 있나: 그 달 1일 기준으로 변환표(adm_map)에 대 봄
④ 관측소가 있는 행정동은 몇 개인가 (전체 행정동 중)
⑤ 요소별로 쓸 수 있는 관측소는 몇 곳인가 (가용성 표)
"""
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)
TZ = "Asia/Seoul"
NUM = ["TA", "WD", "WS", "RN_DAY", "RN_HR1", "HM", "PA", "PS"]

stn = pd.read_parquet("data/bronze/stn_raw.parquet")
stn = pd.DataFrame({
    "STN": stn["STN_ID"].astype("int32"),
    "snap": stn["src_file"].str[8:20],                              # stn_aws_YYYYMMDDHHMI.txt
    "LON": pd.to_numeric(stn["LON"], errors="coerce"),
    "LAT": pd.to_numeric(stn["LAT"], errors="coerce"),
    "HT": pd.to_numeric(stn["HT"], errors="coerce"),
    "LAW_ID": stn["LAW_ID"].str.strip(),
    "STN_KO": stn["STN_KO"].str.strip(),
    "LAW_ADDR": stn["LAW_ADDR"].str.strip(),
}).sort_values(["STN", "snap"]).reset_index(drop=True)
first = stn[stn.snap.str.endswith("010000")].assign(month=lambda x: x.snap.str[:6])   # 매달 1일 지점정보
av = pd.read_parquet("data/silver/aws_availability.parquet")
adm = pd.read_parquet("data/silver/adm_map.parquet")
print(f"지점정보 {stn.snap.nunique()}개 시점, 관측소 {stn.STN.nunique()}곳 / 기상 자료 관측소 {av.station_id.nunique()}곳")

# ================================================================ ① 지점정보와 기상 자료가 맞나
print("\n=== ① 달마다 관측소 수 ===")
data = av.groupby(["station_id", "month"]).n_rows.max().reset_index()
data = data[data.n_rows > 0].assign(STN=lambda x: x.station_id.astype("int32"))
listed = first[["STN", "month"]].assign(listed=True)
both = data[["STN", "month"]].merge(listed, on=["STN", "month"], how="outer", indicator=True)
tab = both.groupby("month")["_merge"].value_counts().unstack(fill_value=0)
tab.columns = [{"both": "둘 다", "left_only": "자료만 있음", "right_only": "지점정보만 있음"}[c] for c in tab.columns]
print(tab.to_string())
only_data = both[both._merge == "left_only"]
print(f"자료는 있는데 그 달 1일 지점정보에 없는 관측소×월: {len(only_data)} (월 중간 신설로 보임) 예:",
      only_data.head(8)[["STN", "month"]].values.tolist())

# ================================================================ ② 관측소가 옮겨졌나
print("\n=== ② 지점정보 시점 사이의 변화 (같은 관측소, 바로 앞 시점과 비교) ===")
p = stn.groupby("STN").shift()
same = stn.STN == stn.STN.shift()
lat1, lat2 = np.radians(p.LAT), np.radians(stn.LAT)
dx = np.radians(stn.LON - p.LON) * np.cos((lat1 + lat2) / 2)
km = 6371 * np.hypot(dx, lat2 - lat1)
ch = pd.DataFrame({"STN": stn.STN, "이름": stn.STN_KO, "앞 시점": p.snap, "시점": stn.snap, "이동_km": km.round(3),
                   "높이변화_m": (stn.HT - p.HT).round(1), "LAW_ID변경": stn.LAW_ID != p.LAW_ID,
                   "이름변경": stn.STN_KO != p.STN_KO})[same]
moved = ch["이동_km"] > 0
print(f"비교한 쌍 {len(ch):,} / 좌표가 바뀜 {int(moved.sum())} (0.1 km 이상 {int((ch['이동_km'] >= 0.1).sum())}, 1 km 이상 {int((ch['이동_km'] >= 1).sum())})"
      f" / 높이 5 m 이상 변화 {int((ch['높이변화_m'].abs() >= 5).sum())} / LAW_ID 변경 {int(ch['LAW_ID변경'].sum())} / 이름 변경 {int(ch['이름변경'].sum())}")
print(f"  LAW_ID만 바뀌고 좌표는 그대로: {int((ch['LAW_ID변경'] & ~moved).sum())}  /  0.1 km 이상 옮겼는데 LAW_ID는 그대로: {int((~ch['LAW_ID변경'] & (ch['이동_km'] >= 0.1)).sum())}")
print("  0.1 km 이상 옮긴 경우 (큰 순 15):")
print(ch[ch["이동_km"] >= 0.1].sort_values("이동_km", ascending=False).head(15).to_string(index=False))

# ================================================================ ③ LAW_ID로 행정동을 정할 수 있나
print("\n=== ③ LAW_ID → 행정동 (그 달 1일에 유효한 읍면동 수준 관계) ===")
first = first.assign(day=pd.to_datetime(first.month + "01", format="%Y%m%d").dt.tz_localize(TZ))
emd = adm[adm.admin_level == "emd"][["legal_dong_code", "admin_dong_code", "admin_dong_name", "valid_from", "valid_to"]]
m = first[["STN", "month", "day", "LAW_ID"]].merge(emd, left_on="LAW_ID", right_on="legal_dong_code", how="left")
ok = (m.valid_from <= m.day) & (m.valid_to.isna() | (m.valid_to > m.day))
n_adm = m[ok].groupby(["STN", "month"]).admin_dong_code.nunique()
res = first[["STN", "month", "LAW_ID", "LAW_ADDR"]].merge(n_adm.rename("n").reset_index(), on=["STN", "month"], how="left").fillna({"n": 0})
sigungu = res.LAW_ID.str[5:8] == "000"
res["판정"] = np.select([sigungu, res.n == 1, res.n >= 2], ["시군구까지만", "하나로 정해짐", "모호(2개 이상)"], "그날 유효한 줄 없음")
print(res["판정"].value_counts().to_string())
print("달별:", res.groupby("month")["판정"].apply(lambda x: (x == "하나로 정해짐").mean().round(3)).iloc[[0, 10, 11, -1]].to_dict(), "(하나로 정해진 비율)")
for k in ("그날 유효한 줄 없음", "모호(2개 이상)", "시군구까지만"):
    ex = res[res["판정"] == k].drop_duplicates("STN").head(5)
    print(f"  [{k}] 관측소 {res[res['판정'] == k].STN.nunique()}곳. 예:", ex[["STN", "month", "LAW_ID", "LAW_ADDR"]].values.tolist())

# ================================================================ ④ 관측소가 있는 행정동 수
print("\n=== ④ 관측소가 있는 행정동 (LAW_ID로 하나로 정해진 경우만, 참고용) ===")
one = m[ok].merge(res[res["판정"] == "하나로 정해짐"][["STN", "month"]], on=["STN", "month"])
for mo in sorted(first.month.unique())[::11] + [sorted(first.month.unique())[-1]]:
    d = pd.Timestamp(mo + "01", tz=TZ)
    total = emd[(emd.valid_from <= d) & (emd.valid_to.isna() | (emd.valid_to > d))].admin_dong_code.nunique()
    has = one[one.month == mo].admin_dong_code.nunique()
    print(f"  {mo}: 행정동 {total:,}개 중 관측소가 있는 곳 {has:,}개 ({has / total:.1%})")

# ================================================================ ⑤ 요소별로 쓸 수 있는 관측소 수
print("\n=== ⑤ 요소별로 그 달 값이 50% 이상 남은 관측소 수 (24개월 중간값 / 최소) ===")
use = av[av.value_share >= 0.5].groupby(["element", "month"]).station_id.nunique().unstack()
print(pd.DataFrame({"중간값": use.median(axis=1), "최소": use.min(axis=1)}).reindex(NUM).astype(int).to_string())