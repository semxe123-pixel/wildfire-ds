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
# 열 이름은 설명서 표준명 (admin_dong_code, admin_dong_name, bronze_file). 법정동 쪽은 같은 방식으로 legal_dong_*
m = pd.DataFrame({
    "admin_dong_code": mix["행정동코드"].astype(str),
    "legal_dong_code": mix["법정동코드"].astype(str),
    "sido_name": mix["시도명"],
    "sigungu_name": mix["시군구명"],
    "admin_dong_name": mix["읍면동명"],          # 행정동 이름
    "legal_dong_name": mix["동리명"],            # 법정동(리) 이름
    "valid_from": kst_date(mix["생성일자"]),
    "valid_to": kst_date(mix["말소일자"]),       # 없으면 NaT = 지금도 유효
    "bronze_file": "adm/" + mix["src_file"],
}).reset_index(drop=True)
print(f"KIKmix({SNAPSHOT}) {len(m):,}행 / 생성일자 변환 실패 {m.valid_from.isna().sum()} / 말소일자 없음 {m.valid_to.isna().sum():,}")


# 2) 코드 자릿수로 수준 구분. 코드 = 시도(2) 시군구(3) 읍면동(3) 리(2)
def adm_level(c):
    return "sido" if c[2:5] == "000" else "sigungu" if c[5:8] == "000" else "emd"


def law_level(c):
    return "sido" if c[2:5] == "000" else "sigungu" if c[5:8] == "000" else "emd" if c[8:10] == "00" else "ri"


m["admin_level"] = m.admin_dong_code.map(adm_level)
m["legal_level"] = m.legal_dong_code.map(law_level)
print("행정동 수준:", m.admin_level.value_counts().to_dict())


# 3) A1: 날짜 d에 유효한 관계
def valid_on(df, d, inclusive_end=False):
    d = day(d)
    alive = df.valid_to.isna() | (df.valid_to >= d if inclusive_end else df.valid_to > d)
    return df[(df.valid_from <= d) & alive]


# 검증 ① A1의 근거: 2026-01-01 유효 관계 vs 20260101 일반판
plain = raw[(raw["table"] == "KIKmix") & (raw["snapshot"] == "jscode20260101")]
ref = set(zip(plain["행정동코드"], plain["법정동코드"]))
for inc in (False, True):
    got = set(zip(*[valid_on(m, "2026-01-01", inc)[c] for c in ("admin_dong_code", "legal_dong_code")]))
    print(f"[검증①] 말소일자 {'포함' if inc else '미포함'}: 유효 {len(got):,} / 일반판 {len(ref):,} "
          f"/ 말소판에만 {len(got - ref)} / 일반판에만 {len(ref - got)}")

# 검증 ② A3의 근거: 법정동 하나가 행정동 여러 개에 걸치는 경우
v = valid_on(m, "2025-03-22")
v = v[v.admin_level == "emd"]
multi = v.groupby("legal_dong_code").admin_dong_code.nunique()
print(f"[검증②] 2025-03-22 법정동 {len(multi):,}개 중 행정동 2개 이상 {int((multi > 1).sum())}개")


# 4) 기간 중 행정동 관계 변경 목록 (읍면동 수준, 생성=말소 같은 날인 행 제외)
emd = m[(m.admin_level == "emd") & (m.valid_from != m.valid_to)]
lo, hi = PERIOD
starts = emd[(emd.valid_from > lo) & (emd.valid_from <= hi)].assign(change="start", date=lambda x: x.valid_from)
ends = emd[(emd.valid_to > lo) & (emd.valid_to <= hi)].assign(change="end", date=lambda x: x.valid_to)
ch = pd.concat([starts, ends], ignore_index=True)

# 코드만 바뀐 것인가: 같은 날 반대쪽(start↔end)에 이름이 같은 관계가 있으면 코드만 바뀐 것
# False = 이름도 바뀜 (행정동 이름 변경, 분동·합동, 관할 이동 등. 어느 것인지는 목록을 보고 판단)
ch["name_key"] = list(zip(ch.sido_name.str.replace("전라북도", "전북특별자치도"), ch.sigungu_name, ch.admin_dong_name, ch.legal_dong_name))
other = {(r.date, r.change, r.name_key) for r in ch.itertuples()}
flip = {"start": "end", "end": "start"}
ch["code_only"] = [(r.date, flip[r.change], r.name_key) in other for r in ch.itertuples()]
ch = ch.drop(columns="name_key").sort_values(["date", "change", "admin_dong_code", "legal_dong_code"])
ch["date"] = ch["date"].dt.strftime("%Y-%m-%d")
cols = ["date", "change", "code_only", "sido_name", "sigungu_name", "admin_dong_code", "admin_dong_name", "legal_dong_code", "legal_dong_name"]
ch[cols].to_csv("data/silver/adm_changes_2024_2025.csv", index=False, encoding="utf-8-sig")

summary = ch.groupby("date").agg(관계_시작=("change", lambda x: (x == "start").sum()),
                                 관계_끝=("change", lambda x: (x == "end").sum()),
                                 이름도_바뀜=("code_only", lambda x: (~x).sum()))
print(f"\n[기간 중 변경] {len(ch):,}행 → data/silver/adm_changes_2024_2025.csv")
print(summary.to_string())

# 5) 경계 기준일 사이에 행정동 목록(이름 기준)이 달라졌나
def adm_names(d):
    x = valid_on(emd, d)
    return set(zip(x.sido_name.str.replace("전라북도", "전북특별자치도"), x.sigungu_name, x.admin_dong_name))

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