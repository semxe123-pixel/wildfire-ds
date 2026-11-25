"""Silver 8: 검증 보고 → data/silver/silver_report.md (+ 화면 출력)

Silver 표들이 '규칙대로 만들어졌는지'와 'Gold에서 서로 이어지는지'를 숫자로 확인한다. 표를 고치지 않음 (읽기만).

검사 목록 (각 항목: 통과 / 주의 / 실패)
  S0 파일 목록: 표마다 행 수·열 수
  S1 변환표 adm_map: 날짜별 유효한 행정동 수, 한 법정동이 같은 날 행정동 둘에 속하는 경우(법정동 하나가 여러 행정동에 걸친 경우는 정상 → 개수만 보고)
  S2 기상 aws_silver
       키 중복 없음(관측소 + 관측 시각) · 관측 시각이 2024-01-01 00시 ~ 2025-12-31 23시 안 · 정시만
       available_at = observed_at + 1시간
       판정과 값의 일치: 판정 1(결측)·3(제거)이면 값이 비어 있고, 0(정상)·2(확인 필요)면 값이 있음
       요소별 판정 비율
  S3 가용성 aws_availability: 관측소 × 월 × 요소의 n_rows 합 = 기상 표 행 수
  S4 관측소 소재지 stn_adm: 관측소 × 월마다 날짜 범위가 빈틈·겹침 없이 한 달을 덮음
  S5 대표점 adm_point: 매월 1일과 행정동 코드가 바뀌는 날마다, 그날 유효한 행정동(출장소 제외)이 모두 대표점을 가짐
  S6 후보 관측소 stn_assign: 순위가 1부터 빈틈 없음 · 거리 30 km 이하 · 후보 없는 (행정동, 기간, 요소) 수
       후보 관측소가 그 달 그 요소의 값을 가진 관측소인지(D2: value_share > 0)
  S7 산불 Label fire_label: 사건 번호 중복 없음 · 주소 매핑·소방 연결 결과 · 추정 매핑 목록
       행정동 코드가 발생 시각의 대표점·후보 관측소와 이어지는지
  S8 미리 보기: 발생 시각 기준 '이미 받을 수 있었던 최신 관측'(available_at ≤ 발생 시각의 정시)을
       후보 관측소에서 가까운 순서로 찾았을 때 값이 있는 사건 비율 (Gold D4 규칙 미리 보기)

입력  data/silver/*.parquet
출력  data/silver/silver_report.md
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)
TZ = "Asia/Seoul"
S = "data/silver"
T0, T1 = pd.Timestamp("2024-01-01 00:00", tz=TZ), pd.Timestamp("2025-12-31 23:00", tz=TZ)
FAR = pd.Timestamp("2100-01-01", tz=TZ)
ELEM = ["TA", "WD", "WS", "RN_DAY", "RN_HR1", "HM", "PA", "PS"]
lines, summary = [], []


def md(s=""):
    lines.append(s)
    print(s)


def check(code, name, ok, detail, warn=False):
    mark = "통과" if ok else ("주의" if warn else "실패")
    summary.append((code, name, mark, detail))
    md(f"- **[{mark}] {code} {name}**: {detail}")


def table(df):
    try:
        md(df.to_markdown(index=False))                 # tabulate 패키지가 있으면 표로
    except ImportError:
        md("```\n" + df.to_string(index=False) + "\n```")
    md()


# ================================================================ S0 파일 목록
md("# Silver 검증 보고")
md(f"\n생성: {pd.Timestamp.now(tz=TZ):%Y-%m-%d %H:%M} KST · 기간 {T0:%Y-%m-%d %H:%M} ~ {T1:%Y-%m-%d %H:%M} (17,544시간)\n")
md("## S0 표 목록\n")
rows = []
for f in sorted(glob.glob(f"{S}/*.parquet")):
    if os.path.basename(f).startswith("aws_silver_"):
        continue
    d = pd.read_parquet(f)
    rows.append({"표": os.path.basename(f), "행": f"{len(d):,}", "열": d.shape[1]})
aws_files = sorted(glob.glob(f"{S}/aws_silver_*.parquet"))
table(pd.DataFrame(rows + [{"표": f"aws_silver_YYYYMM ({len(aws_files)}개)", "행": "S2 참고", "열": ""}]))

# ================================================================ S1 변환표
md("## S1 변환표 adm_map\n")
adm = pd.read_parquet(f"{S}/adm_map.parquet")
emd = adm[(adm.admin_level == "emd")].copy()
emd["vt"] = emd.valid_to.fillna(FAR)
branch = emd.admin_dong_name.astype(str).str.contains("출장소")
days = [pd.Timestamp(x, tz=TZ) for x in ("2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01", "2025-12-31")]
cnt = [{"날짜": f"{d:%Y-%m-%d}",
        "유효 행정동(출장소 제외)": emd[~branch & (emd.valid_from <= d) & (emd.vt > d)].admin_dong_code.nunique(),
        "법정동이 행정동 2곳 이상에 걸침": int((emd[~branch & (emd.valid_from <= d) & (emd.vt > d)]
                                         .groupby("legal_dong_code").admin_dong_code.nunique() > 1).sum())} for d in days]
table(pd.DataFrame(cnt))
same_day = int((emd.valid_from == emd.vt).sum())
bad = int((emd.valid_from > emd.vt).sum())
check("S1-1", "유효기간 시작 ≤ 끝", bad == 0,
      f"시작 > 끝 {bad}행(원본 이상, A1 조건 '시작 ≤ d < 끝'을 어느 날에도 만족하지 않아 쓰이지 않음) / "
      f"같은 날 생성·말소 {same_day}행(원본 그대로, 쓰이지 않음)", warn=True)
if bad:
    md("\n시작 > 끝인 원본 행\n")
    table(emd[emd.valid_from > emd.vt][["admin_dong_code", "admin_dong_name", "legal_dong_code", "legal_dong_name",
                                        "valid_from", "valid_to"]].astype(str))

# ================================================================ S2·S3 기상
md("\n## S2 기상 aws_silver\n")
tot, dup, out_rng, not_hour, avail_bad = 0, 0, 0, 0, 0
qc = {e: np.zeros(4, dtype=np.int64) for e in ELEM}
mismatch = {e: 0 for e in ELEM}
per_month, seen = [], {}
for f in aws_files:
    d = pd.read_parquet(f)
    tot += len(d)
    dup += int(d.duplicated(["station_id", "observed_at"]).sum())
    out_rng += int(((d.observed_at < T0) | (d.observed_at > T1)).sum())
    not_hour += int((d.observed_at != d.observed_at.dt.floor("h")).sum())
    avail_bad += int((d.available_at - d.observed_at != pd.Timedelta(hours=1)).sum())
    for e in ELEM:
        q = d[f"qc_{e}"].to_numpy()
        qc[e] += np.bincount(q.astype(np.int64), minlength=4)[:4]
        isnull = d[e].isna().to_numpy()
        mismatch[e] += int((np.isin(q, [1, 3]) & ~isnull).sum() + (np.isin(q, [0, 2]) & isnull).sum())
    seen[os.path.basename(f)[11:17]] = d.groupby("station_id").size()
    per_month.append({"월": os.path.basename(f)[11:17], "행": len(d), "관측소": d.station_id.nunique()})
md(f"전체 {tot:,}행, 월 파일 {len(aws_files)}개\n")
check("S2-1", "키 중복 없음 (관측소 + 관측 시각)", dup == 0, f"중복 {dup}")
check("S2-2", "관측 시각이 기간 안·정시", out_rng == 0 and not_hour == 0, f"기간 밖 {out_rng} / 정시 아님 {not_hour}")
check("S2-3", "available_at = observed_at + 1시간", avail_bad == 0, f"어긋남 {avail_bad}")
check("S2-4", "판정과 값 일치 (1·3 → 빈 값, 0·2 → 값 있음)", sum(mismatch.values()) == 0,
      ", ".join(f"{e} {n}" for e, n in mismatch.items()))
qt = pd.DataFrame([{"요소": e, "정상(0)": f"{v[0] / v.sum():.2%}", "결측(1)": f"{v[1] / v.sum():.2%}",
                    "확인 필요(2)": f"{v[2]:,}", "제거(3)": f"{v[3]:,}"} for e, v in qc.items()])
md("\n요소별 판정\n")
table(qt)

md("## S3 가용성 aws_availability\n")
av = pd.read_parquet(f"{S}/aws_availability.parquet")
s3 = av.groupby("element").n_rows.sum()
ok3 = all(int(s3.get(e, -1)) == tot for e in ELEM)
check("S3-1", "요소별 n_rows 합 = 기상 표 행 수", ok3, ", ".join(f"{e} {int(s3.get(e, 0)):,}" for e in ELEM))
cmp = av[av.element == "TA"].set_index(["month", "station_id"]).n_rows
real = pd.concat(seen, names=["month", "station_id"])
diff = int((cmp.reindex(real.index).fillna(-1) != real).sum())
check("S3-2", "관측소 × 월 행 수가 기상 표와 같음", diff == 0, f"다른 칸 {diff}")

# ================================================================ S4 관측소 소재지
md("\n## S4 관측소 소재지 stn_adm\n")
sa = pd.read_parquet(f"{S}/stn_adm.parquet").sort_values(["station_id", "month", "valid_from"])
g = sa.groupby(["station_id", "month"])
m0 = pd.to_datetime(sa.month + "01").dt.tz_localize(TZ)
first_ok = (g.valid_from.first() == m0.groupby([sa.station_id, sa.month]).first())
last_ok = (g.valid_to.last() == (m0 + pd.offsets.MonthBegin(1)).groupby([sa.station_id, sa.month]).first())
chain = sa.groupby(["station_id", "month"]).apply(lambda x: bool((x.valid_from.iloc[1:].values == x.valid_to.iloc[:-1].values).all()),
                                                    include_groups=False)
gaps = int((~first_ok).sum() + (~last_ok).sum() + (~chain).sum())
check("S4-1", "관측소 × 월 날짜 범위가 빈틈·겹침 없이 한 달을 덮음", gaps == 0, f"문제 {gaps} / 관측소 × 월 {g.ngroups:,}")
md(f"\n판정 방법 {sa.spatial_mapping_method.value_counts().to_dict()} · 코드 {sa.code_mapping.value_counts().to_dict()} · "
   f"추정 {int(sa.is_estimated.sum())}행\n")

# ================================================================ S5 대표점
md("## S5 대표점 adm_point\n")
ap = pd.read_parquet(f"{S}/adm_point.parquet")
change = sorted({d for d in set(emd.valid_from.dropna()) | set(emd.valid_to.dropna()) if T0 <= d <= T1})
check_days = sorted(set(pd.date_range(T0, T1, freq="MS")) | set(change))
miss = []
for d in check_days:
    need = set(emd[~branch & (emd.valid_from <= d) & (emd.vt > d)].admin_dong_code)
    have = set(ap[(ap.valid_from <= d) & (ap.valid_to > d)].admin_dong_code)
    if need - have:
        miss.append((f"{d:%Y-%m-%d}", len(need - have), sorted(need - have)[:5]))
check("S5-1", "확인한 날마다 유효 행정동이 모두 대표점을 가짐", not miss,
      f"확인한 날 {len(check_days)}일 / 빠진 날 {len(miss)}" + (f" 예: {miss[:3]}" if miss else ""))
md(f"\n대표점 방법 {ap.centroid_method.value_counts().to_dict()} · 구역 출처 {ap.polygon_source.value_counts().to_dict()} · "
   f"잘라낸 구역(R4) {int(ap.trimmed.sum())}행\n")

# ================================================================ S6 후보 관측소
md("## S6 후보 관측소 stn_assign\n")
st = pd.read_parquet(f"{S}/stn_assign.parquet")
key = ["admin_dong_code", "valid_from", "element"]
rk = st.groupby(key)["rank"].agg(["min", "max", "size"])
check("S6-1", "순위가 1부터 빈틈 없음", bool(((rk["min"] == 1) & (rk["max"] == rk["size"])).all()),
      f"(행정동, 기간, 요소) {len(rk):,}개")
check("S6-2", "거리 30 km 이하", bool((st.distance_km <= 30).all()), f"최대 {st.distance_km.max():.2f} km")
# 대표점의 각 기간 가운데 시각에 유효한 후보가 요소마다 있는지 (기간 경계가 달라도 맞게 비교)
grid = ap[["admin_dong_code", "valid_from", "valid_to"]].drop_duplicates()
grid["mid"] = grid.valid_from + (grid.valid_to - grid.valid_from) / 2
hit = grid[["admin_dong_code", "mid"]].merge(st[["admin_dong_code", "valid_from", "valid_to", "element"]].drop_duplicates(),
                                             on="admin_dong_code")
hit = hit[(hit.valid_from <= hit.mid) & (hit.valid_to > hit.mid)][["admin_dong_code", "mid", "element"]].drop_duplicates()
full = grid[["admin_dong_code", "mid"]].merge(pd.DataFrame({"element": ELEM}), how="cross")
nocand = full.merge(hit, on=["admin_dong_code", "mid", "element"], how="left", indicator=True)
nocand = nocand[nocand._merge == "left_only"]
grid = full
check("S6-3", "후보 없는 (행정동, 기간, 요소)", len(nocand) == 0,
      f"{len(nocand):,}개 / 전체 {len(grid):,}" + (f" · 요소별 {nocand.element.value_counts().to_dict()}" if len(nocand) else ""),
      warn=True)
vs = av.set_index(["station_id", "month", "element"]).value_share
st_vs = vs.reindex(pd.MultiIndex.from_frame(st[["station_id", "month", "element"]])).to_numpy()
check("S6-4", "후보 관측소가 그 달 그 요소의 값을 가짐 (D2)", bool(np.nan_to_num(st_vs) .min() > 0),
      f"값 비율 0 또는 없음 {int((np.nan_to_num(st_vs) <= 0).sum())}행")
md(f"\n1순위 거리 중앙값 {st[st['rank'] == 1].distance_km.median():.2f} km · 후보 수 분포 "
   f"{rk['size'].value_counts().sort_index().to_dict()}\n")

# ================================================================ S7 산불 Label
md("## S7 산불 Label fire_label\n")
fl = pd.read_parquet(f"{S}/fire_label.parquet")
check("S7-1", "사건 번호 중복 없음", fl.fire_id.is_unique, f"{len(fl)}건")
check("S7-2", "소방 기록 1건이 사건 1건에만 연결", not fl.fire119_wrinv_no.dropna().duplicated().any(),
      f"연결 {int(fl.fire119_linked.sum())}/{len(fl)}")
st7 = fl.addr_mapping_status.value_counts().to_dict()
check("S7-3", "행정동 확정", st7.get("confirmed", 0) == len(fl),
      f"{st7} / 방법 {fl.addr_mapping_method.replace('', '(없음)').value_counts().to_dict()}", warn=True)
md("\n주소를 고쳐서 찾은 사건(추정 매핑) — 눈으로 확인할 목록\n")
est = fl[fl.addr_is_estimated | (fl.addr_mapping_method == "geocode")]
table(est[["fire_id", "sido_raw", "sigungu_raw", "emd_raw", "ri_dong_raw", "bunji_raw", "addr_mapping_method",
           "admin_dong_name", "sigungu_name"]])
md("행정동을 못 정한 사건\n")
table(fl[fl.admin_dong_code.isna()][["fire_id", "sido_raw", "sigungu_raw", "emd_raw", "ri_dong_raw", "bunji_raw",
                                     "addr_text_status", "damage_area_ha"]])
fc = fl[fl.admin_dong_code.notna()]
lk_p = fc.merge(ap[["admin_dong_code", "valid_from", "valid_to"]], on="admin_dong_code")
lk_p = lk_p[(lk_p.valid_from <= lk_p.ignition_at) & (lk_p.valid_to > lk_p.ignition_at)]
miss7 = sorted(set(fc.fire_id) - set(lk_p.fire_id))
check("S7-4", "행정동이 발생 시각의 대표점과 이어짐", not miss7, f"{lk_p.fire_id.nunique()}/{len(fc)}"
      + (" / 안 이어진 사건: " + ", ".join(f"{x}({fc.set_index('fire_id').at[x, 'admin_dong_name']}, "
                                         f"{fc.set_index('fire_id').at[x, 'admin_dong_code']})" for x in miss7) if miss7 else ""))
lk_s = fc[["fire_id", "admin_dong_code", "ignition_at"]].merge(st, on="admin_dong_code")
lk_s = lk_s[(lk_s.valid_from <= lk_s.ignition_at) & (lk_s.valid_to > lk_s.ignition_at)]
per_e = lk_s.groupby("element").fire_id.nunique().reindex(ELEM).fillna(0).astype(int)
check("S7-5", "행정동이 발생 시각의 후보 관측소와 이어짐 (요소별 사건 수)", bool((per_e == len(fc)).all()),
      ", ".join(f"{e} {n}" for e, n in per_e.items()), warn=True)

# ================================================================ S8 미리 보기 (D4)
md("\n## S8 Gold 미리 보기: 발생 시각에 받을 수 있었던 최신 관측\n")
md("발생 시각의 정시를 기준 시각으로, available_at ≤ 기준 시각인 최신 관측(= 기준 시각 1시간 전 관측)을 "
   "후보 관측소 순위대로 찾아 판정 0·2이고 값이 있는 첫 관측소를 씀 (D4)\n")
lk_s = lk_s.assign(ref=lk_s.ignition_at.dt.floor("h"))
lk_s["obs"] = lk_s.ref - pd.Timedelta(hours=1)
lk_s = lk_s[lk_s.obs >= T0]
need = lk_s.assign(m=lk_s.obs.dt.strftime("%Y%m"))
vals = []
for mo, part in need.groupby("m"):
    f = f"{S}/aws_silver_{mo}.parquet"
    if not os.path.exists(f):
        continue
    d = pd.read_parquet(f, columns=["observed_at", "station_id"] + ELEM + [f"qc_{e}" for e in ELEM])
    d = d[d.station_id.isin(part.station_id.unique()) & d.observed_at.isin(part.obs.unique())]
    long = []
    for e in ELEM:
        x = d[["observed_at", "station_id", e, f"qc_{e}"]].rename(columns={e: "value", f"qc_{e}": "qc"})
        long.append(x.assign(element=e))
    vals.append(pd.concat(long))
pv = pd.concat(vals) if vals else pd.DataFrame(columns=["observed_at", "station_id", "value", "qc", "element"])
j = lk_s.merge(pv, left_on=["obs", "station_id", "element"], right_on=["observed_at", "station_id", "element"], how="left")
j["usable"] = j.value.notna() & j.qc.isin([0, 2])
best = j[j.usable].sort_values("rank").drop_duplicates(["fire_id", "element"])
rows = []
for e in ELEM:
    b = best[best.element == e]
    n_e = lk_s[lk_s.element == e].fire_id.nunique()
    rows.append({"요소": e, "후보 있는 사건": n_e, "값 찾음": len(b), "비율": f"{len(b) / max(n_e, 1):.1%}",
                 "1순위에서": int((b["rank"] == 1).sum()), "2순위 이하": int((b["rank"] > 1).sum()),
                 "쓴 관측소 거리 중앙값 km": round(float(b.distance_km.median()), 2) if len(b) else np.nan})
table(pd.DataFrame(rows))
md(f"(기준 시각 1시간 전이 기간 시작 전인 사건은 제외: {fc.ignition_at.dt.floor('h').sub(pd.Timedelta(hours=1)).lt(T0).sum()}건)\n")

# ================================================================ 요약
md("## 요약\n")
table(pd.DataFrame(summary, columns=["번호", "검사", "결과", "내용"]))
with open(f"{S}/silver_report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"[저장] {S}/silver_report.md")