import os
import pandas as pd

os.makedirs("data/silver", exist_ok=True)
raw = pd.read_parquet("data/bronze/adm_raw.parquet")

# 1) 20261026 말소판 KIKmix만
mix = raw[(raw["table"] == "KIKmix") & (raw["snapshot"] == "jscode20261026_malso")].copy()
print(f"KIKmix(20261026 말소판) {len(mix):,}행")

# 2) 열 이름 통일 + 날짜형
m = pd.DataFrame({
    "adm_cd":  mix["행정동코드"].astype(str),
    "law_cd":  mix["법정동코드"].astype(str),
    "sido":    mix["시도명"],
    "sigungu": mix["시군구명"],
    "adm_nm":  mix["읍면동명"],          # 행정동 이름
    "law_nm":  mix["동리명"],            # 법정동(리) 이름
    "valid_from": pd.to_datetime(mix["생성일자"], format="%Y%m%d", errors="coerce"),
    "valid_to":   pd.to_datetime(mix["말소일자"], format="%Y%m%d", errors="coerce"),   # 없으면 NaT
    "src_file": mix["src_file"],
})
print("날짜 변환 실패: 생성", m.valid_from.isna().sum(), "/ 말소(원래 빈값 포함)", m.valid_to.isna().sum())

# 3) 코드 자릿수로 수준 구분
#    코드 = 시도(2) 시군구(3) 읍면동(3) 리(2)
def adm_level(c):   # 행정동코드
    return "sido" if c[2:5] == "000" else "sigungu" if c[5:8] == "000" else "emd"
def law_level(c):   # 법정동코드
    return "sido" if c[2:5] == "000" else "sigungu" if c[5:8] == "000" else "emd" if c[8:10] == "00" else "ri"
m["adm_level"] = m.adm_cd.map(adm_level)
m["law_level"] = m.law_cd.map(law_level)
print("\n행정동 수준:", m.adm_level.value_counts().to_dict())
print("법정동 수준:", m.law_level.value_counts().to_dict())

# 4) 특정 날짜에 유효한 관계
def valid_on(df, d, inclusive_end):
    d = pd.Timestamp(d)
    alive = df.valid_to.isna() | (df.valid_to >= d if inclusive_end else df.valid_to > d)
    return df[(df.valid_from <= d) & alive]

# 검증 ①: 2026-01-01 유효 관계 == 20260101 일반판(21,816행)?
plain = raw[(raw["table"] == "KIKmix") & (raw["snapshot"] == "jscode20260101")]
ref = set(zip(plain["행정동코드"], plain["법정동코드"]))
for inc in (True, False):
    v = valid_on(m, "2026-01-01", inc)
    got = set(zip(v.adm_cd, v.law_cd))
    print(f"\n[검증①] 말소일자 {'포함' if inc else '미포함'} 기준: 유효 {len(v):,}행 / 일반판 {len(ref):,}행 "
          f"/ 일치 {len(got & ref):,} / 말소판에만 {len(got - ref)} / 일반판에만 {len(ref - got)}")

# 검증 ②: 2025-03-22 기준, 읍면동 수준 행정동에서 법정동 하나 → 행정동 여러 개?
v = valid_on(m, "2025-03-22", inclusive_end=False)
v = v[v.adm_level == "emd"]
multi = v.groupby("law_cd").adm_cd.nunique()
print(f"\n[검증②] 2025-03-22 유효 · 읍면동 수준 관계 {len(v):,}행, 법정동 {v.law_cd.nunique():,}개")
print("   행정동 1개에만 속한 법정동:", int((multi == 1).sum()), " / 2개 이상:", int((multi > 1).sum()))
print("   2개 이상인 법정동의 수준:", v[v.law_cd.isin(multi[multi > 1].index)].drop_duplicates("law_cd").law_level.value_counts().to_dict())
ex = multi[multi > 1].index[:3]
print("   예시:\n", v[v.law_cd.isin(ex)][["law_cd", "law_nm", "adm_cd", "adm_nm", "sigungu"]].to_string(index=False))
# 검증① 차이 행 기록
v = valid_on(m, "2026-01-01", inclusive_end=False)
got = set(zip(v.adm_cd, v.law_cd))
only_plain = ref - got
print("\n[A4] 일반판에만 있는 관계:", only_plain)
pr = plain[plain.apply(lambda r: (r["행정동코드"], r["법정동코드"]) in only_plain, axis=1)]
print(pr[["행정동코드", "읍면동명", "법정동코드", "동리명", "생성일자", "말소일자"]].to_string(index=False))
mm = m[m.law_cd.isin([l for _, l in only_plain])]
print("   말소판에서 같은 법정동:\n", mm[["adm_cd", "adm_nm", "law_cd", "law_nm", "valid_from", "valid_to"]].to_string(index=False))
# 5) 저장
m.to_parquet("data/silver/adm_map.parquet", index=False)
print(f"\n저장: data/silver/adm_map.parquet {len(m):,}행")