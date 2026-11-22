import os
import pandas as pd

TZ = "Asia/Seoul"                      # 공통 기준: 모든 시각에 한국 시간 표시
SNAPSHOT = "jscode20261026_malso"      # A4: 20261026 말소판 단일
PERIOD = (pd.Timestamp("2024-01-01", tz=TZ), pd.Timestamp("2025-12-31", tz=TZ))
BOUNDARY_DATES = ["2024-01-01", "2024-11-30", "2025-12-31"]   # 기간 시작, 행안부 경계 2건의 기준일

os.makedirs("data/silver", exist_ok=True)
raw = pd.read_parquet("data/bronze/adm_raw.parquet")


def kst_date(s):
    """'20240118' → 2024-01-18 00:00+09:00. 빈 값은 NaT"""
    return pd.to_datetime(s, format="%Y%m%d", errors="coerce").dt.tz_localize(TZ)


def day(d):
    return pd.Timestamp(d, tz=TZ)


# 1) 말소판 KIKmix → 영문 열 이름, 날짜형
mix = raw[(raw["table"] == "KIKmix") & (raw["snapshot"] == SNAPSHOT)]
m = pd.DataFrame({
    "adm_cd": mix["행정동코드"].astype(str),
    "law_cd": mix["법정동코드"].astype(str),
    "sido": mix["시도명"],
    "sigungu": mix["시군구명"],
    "adm_nm": mix["읍면동명"],          # 행정동 이름
    "law_nm": mix["동리명"],            # 법정동(리) 이름
    "valid_from": kst_date(mix["생성일자"]),
    "valid_to": kst_date(mix["말소일자"]),   # 없으면 NaT = 지금도 유효
    "src_file": mix["src_file"],
}).reset_index(drop=True)
print(f"KIKmix({SNAPSHOT}) {len(m):,}행 / 생성일자 변환 실패 {m.valid_from.isna().sum()} / 말소일자 없음 {m.valid_to.isna().sum():,}")


# 2) 코드 자릿수로 수준 구분. 코드 = 시도(2) 시군구(3) 읍면동(3) 리(2)
def adm_level(c):
    return "sido" if c[2:5] == "000" else "sigungu" if c[5:8] == "000" else "emd"


def law_level(c):
    return "sido" if c[2:5] == "000" else "sigungu" if c[5:8] == "000" else "emd" if c[8:10] == "00" else "ri"


m["adm_level"] = m.adm_cd.map(adm_level)
m["law_level"] = m.law_cd.map(law_level)
print("행정동 수준:", m.adm_level.value_counts().to_dict())


# 3) A1: 날짜 d에 유효한 관계
def valid_on(df, d, inclusive_end=False):
    d = day(d)
    alive = df.valid_to.isna() | (df.valid_to >= d if inclusive_end else df.valid_to > d)
    return df[(df.valid_from <= d) & alive]


# 검증 ① A1의 근거: 2026-01-01 유효 관계 vs 20260101 일반판
plain = raw[(raw["table"] == "KIKmix") & (raw["snapshot"] == "jscode20260101")]
ref = set(zip(plain["행정동코드"], plain["법정동코드"]))
for inc in (False, True):
    got = set(zip(*[valid_on(m, "2026-01-01", inc)[c] for c in ("adm_cd", "law_cd")]))
    print(f"[검증①] 말소일자 {'포함' if inc else '미포함'}: 유효 {len(got):,} / 일반판 {len(ref):,} "
          f"/ 말소판에만 {len(got - ref)} / 일반판에만 {len(ref - got)}")

# 검증 ② A3의 근거: 법정동 하나가 행정동 여러 개에 걸치는 경우
v = valid_on(m, "2025-03-22")
v = v[v.adm_level == "emd"]
multi = v.groupby("law_cd").adm_cd.nunique()
print(f"[검증②] 2025-03-22 법정동 {len(multi):,}개 중 행정동 2개 이상 {int((multi > 1).sum())}개")


# 4) 기간 중 행정동 관계 변경 목록 (읍면동 수준, 생성=말소 같은 날인 행 제외)
emd = m[(m.adm_level == "emd") & (m.valid_from != m.valid_to)]
lo, hi = PERIOD
starts = emd[(emd.valid_from > lo) & (emd.valid_from <= hi)].assign(change="start", date=lambda x: x.valid_from)
ends = emd[(emd.valid_to > lo) & (emd.valid_to <= hi)].assign(change="end", date=lambda x: x.valid_to)
ch = pd.concat([starts, ends], ignore_index=True)

# 코드만 바뀐 것인가: 같은 날 반대쪽(start↔end)에 이름이 같은 관계가 있으면 코드만 바뀐 것
# False = 이름도 바뀜 (행정동 이름 변경, 분동·합동, 관할 이동 등. 어느 것인지는 목록을 보고 판단)
ch["name_key"] = list(zip(ch.sido.str.replace("전라북도", "전북특별자치도"), ch.sigungu, ch.adm_nm, ch.law_nm))
other = {(r.date, r.change, r.name_key) for r in ch.itertuples()}
flip = {"start": "end", "end": "start"}
ch["code_only"] = [(r.date, flip[r.change], r.name_key) in other for r in ch.itertuples()]
ch = ch.drop(columns="name_key").sort_values(["date", "change", "adm_cd", "law_cd"])
ch["date"] = ch["date"].dt.strftime("%Y-%m-%d")
cols = ["date", "change", "code_only", "sido", "sigungu", "adm_cd", "adm_nm", "law_cd", "law_nm"]
ch[cols].to_csv("data/silver/adm_changes_2024_2025.csv", index=False, encoding="utf-8-sig")

summary = ch.groupby("date").agg(관계_시작=("change", lambda x: (x == "start").sum()),
                                 관계_끝=("change", lambda x: (x == "end").sum()),
                                 이름도_바뀜=("code_only", lambda x: (~x).sum()))
print(f"\n[기간 중 변경] {len(ch):,}행 → data/silver/adm_changes_2024_2025.csv")
print(summary.to_string())

# 5) 경계 기준일 사이에 행정동 목록(이름 기준)이 달라졌나
def adm_names(d):
    x = valid_on(emd, d)
    return set(zip(x.sido.str.replace("전라북도", "전북특별자치도"), x.sigungu, x.adm_nm))

for d1, d2 in zip(BOUNDARY_DATES, BOUNDARY_DATES[1:]):
    a, b = adm_names(d1), adm_names(d2)
    print(f"\n[{d1} → {d2}] 행정동 {len(a):,} → {len(b):,} / 없어짐 {len(a - b)} / 생김 {len(b - a)}")
    for t in sorted(a - b):
        print("   없어짐:", " ".join(t))
    for t in sorted(b - a):
        print("   생김:  ", " ".join(t))

# 6) 저장
m.to_parquet("data/silver/adm_map.parquet", index=False)
print(f"\n저장: data/silver/adm_map.parquet {len(m):,}행")